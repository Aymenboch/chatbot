import json
import numpy as np
import matplotlib.pyplot as plt
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory # Added
from tqdm import tqdm
import os
import time
import sys
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from modelEval.py (which should be in the same eval_scripts directory)
try:
    from modelEval import calculate_metrics # calculate_metrics is now in modelEval.py
except ImportError as e:
    print(f"Error importing from modelEval.py: {e}")
    print("Ensure modelEval.py is in the same directory (eval_scripts).")
    sys.exit(1)

# Import from app.py (Streamlit app file)
try:
    from chatbot import load_chroma_vector_store, create_custom_prompt
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the project root and contains necessary functions.")
    sys.exit(1)

# Load environment variables
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Configuration from .env
gemini_key = os.getenv("GEMINI_API_KEY")
model_name_env = os.getenv("MODEL_NAME", "gemini-pro") # Default from your app

# Load test data
test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.json')
try:
    with open(test_data_path, 'r', encoding="utf-8") as f:
        test_data = json.load(f)
except FileNotFoundError:
    print(f"Error: test_data.json not found at {test_data_path}")
    sys.exit(1)

def evaluate_temperatures(sample_size=5, question_delay=6, trial_delay=10):
    """Evaluate different temperature settings with rate limiting"""
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # Added 0.9
    all_temp_run_results = []
    
    chroma_db_path_env = os.getenv("CHROMADB_PATH", "vectorDB/chroma_db_cleaned")
    chroma_db_full_path = os.path.join(project_root, chroma_db_path_env)

    if not os.path.exists(chroma_db_full_path):
        print(f"Chroma DB path not found: {chroma_db_full_path}")
        return []

    vectordb = load_chroma_vector_store(chroma_db_full_path)
    prompt_template = create_custom_prompt()

    for temp in tqdm(temperatures, desc="Testing temperatures"):
        print(f"\n--- Evaluating Temperature: {temp} ---")
        current_temp_metrics_list = []
        
        llm = ChatGoogleGenerativeAI(
            model=model_name_env,
            temperature=temp,
            google_api_key=gemini_key,
            max_output_tokens=2048
        )
        
        # Create a new memory instance for each temperature trial to ensure independence
        # AND clear it before each question within that trial.
        memory_for_temp_trial = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=False
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={
                "prompt": prompt_template,
                "memory": memory_for_temp_trial 
            },
            input_key="query"
        )
        
        # Use a subset of test data for temperature tuning
        subset_test_data = test_data[:sample_size]
        
        for i, item in enumerate(subset_test_data):
            print(f"  Temp {temp} - Q {i+1}/{len(subset_test_data)}: {item['question'][:50]}...")
            try:
                memory_for_temp_trial.chat_memory.clear() # Clear history for this specific question

                response_data = qa_chain.invoke({"query": item["question"]})
                response_text = response_data['result']

                metrics = calculate_metrics(response_text, item["reference_answer"])
                current_temp_metrics_list.append(metrics)
                
                if i < len(subset_test_data) - 1: # If not the last question in this temp trial
                    print(f"    Waiting {question_delay}s before next question...")
                    time.sleep(question_delay)

            except Exception as e:
                print(f"    Error at temp {temp} for question '{item['question']}': {str(e)}")
                import traceback
                print(traceback.format_exc())
                # Optionally, wait longer on error or skip remaining questions for this temp
                # time.sleep(30) 
                continue # Skip to next question or item
        
        if current_temp_metrics_list: # Only if some questions were successful for this temp
            avg_metrics = {
                "temperature": temp,
                "avg_bleu": np.mean([r["bleu_score"] for r in current_temp_metrics_list]),
                "avg_rouge1": np.mean([r["rouge1"] for r in current_temp_metrics_list]),
                "avg_rougeL": np.mean([r["rougeL"] for r in current_temp_metrics_list]),
                "avg_bert_f1": np.mean([r["bert_f1"] for r in current_temp_metrics_list]),
                "avg_cosine_similarity": np.mean([r["cosine_similarity"] for r in current_temp_metrics_list]),
                "sample_responses": [r["response_raw"] for r in current_temp_metrics_list[:1]], # Get one raw response
                "num_successful_questions": len(current_temp_metrics_list)
            }
            all_temp_run_results.append(avg_metrics)
        else:
            all_temp_run_results.append({
                "temperature": temp,
                "error": "No successful questions for this temperature.",
                "num_successful_questions": 0
            })

        if temp != temperatures[-1]:  # Don't delay after the last temperature
            print(f"\nWaiting {trial_delay}s before next temperature trial...")
            time.sleep(trial_delay)
    
    return all_temp_run_results

def plot_temperature_results(results_to_plot):
    """Generate visualization with error handling"""
    valid_results_for_plot = [r for r in results_to_plot if "error" not in r and r["num_successful_questions"] > 0]
    
    if not valid_results_for_plot:
        print("No valid results with successful questions to plot.")
        return
    
    plt.figure(figsize=(12, 7)) # Increased figure size
    
    metrics_to_plot = [
        ('avg_bleu', 'Avg BLEU', 'b'),
        ('avg_rouge1', 'Avg ROUGE-1 F1', 'r'),
        ('avg_rougeL', 'Avg ROUGE-L F1', 'm'), # Added ROUGE-L
        ('avg_bert_f1', 'Avg BERTScore F1', 'g'),
        ('avg_cosine_similarity', 'Avg Cosine Sim.', 'c') # Added Cosine Sim
    ]
    
    temperatures_for_plot = [r["temperature"] for r in valid_results_for_plot]

    for metric_key, label, color_code in metrics_to_plot:
        metric_values = [r[metric_key] for r in valid_results_for_plot]
        plt.plot(
            temperatures_for_plot,
            metric_values,
            marker='o', # Changed for better visibility
            linestyle='-',
            color=color_code,
            label=label
        )
    
    plt.title("LLM Performance vs. Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Average Score")
    plt.xticks(temperatures_for_plot) # Ensure all tested temperatures are shown as ticks
    plt.legend(loc='best') # Auto-find best location for legend
    plt.grid(True, linestyle='--', alpha=0.7) # Nicer grid
    plt.tight_layout() # Adjust layout
    
    plot_file_path = os.path.join(os.path.dirname(__file__), 'temperature_impact_results.png')
    plt.savefig(plot_file_path)
    print(f"Temperature impact plot saved to {plot_file_path}")
    plt.close()

if __name__ == "__main__":
    # Run temperature evaluation
    print("Starting temperature evaluation...")
    # Consider reducing sample_size if API calls are an issue, or increase delays
    temperature_evaluation_results = evaluate_temperatures(sample_size=3, question_delay=10, trial_delay=20) 
    
    results_file_path = os.path.join(os.path.dirname(__file__), 'temperature_evaluation_results.json')
    if temperature_evaluation_results:
        with open(results_file_path, 'w', encoding="utf-8") as f:
            json.dump(temperature_evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"\nTemperature evaluation results saved to {results_file_path}")
        
        plot_temperature_results(temperature_evaluation_results)
        
        # Find optimal temperature based on a primary metric (e.g., BERT F1 or a composite score)
        valid_results_for_optim = [r for r in temperature_evaluation_results if "error" not in r and r["num_successful_questions"] > 0]
        if valid_results_for_optim:
            optimal_temp_data = max(valid_results_for_optim, key=lambda x: x["avg_bert_f1"]) # Example: optimizing for BERT F1
            print(f"\nOptimal temperature based on Avg BERT F1: {optimal_temp_data['temperature']}")
            print(f"  Avg BLEU: {optimal_temp_data['avg_bleu']:.4f}")
            print(f"  Avg ROUGE-1 F1: {optimal_temp_data['avg_rouge1']:.4f}")
            print(f"  Avg ROUGE-L F1: {optimal_temp_data['avg_rougeL']:.4f}")
            print(f"  Avg BERTScore F1: {optimal_temp_data['avg_bert_f1']:.4f}")
            print(f"  Avg Cosine Similarity: {optimal_temp_data['avg_cosine_similarity']:.4f}")
            print(f"  Sample response: {optimal_temp_data['sample_responses'][0] if optimal_temp_data['sample_responses'] else 'N/A'}")
        else:
            print("Could not determine optimal temperature due to lack of valid results.")
            
    else:
        print("No results generated from temperature evaluation.")