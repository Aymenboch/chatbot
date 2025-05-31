import json
import numpy as np
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory # Added
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # Using sacrebleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import BERTScorer
import os
from dotenv import load_dotenv
import sys
import sacrebleu
from unidecode import unidecode
import time # For potential delays if needed

# Add project root to Python path to allow importing from app.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Assuming your Streamlit app file is named app.py in the project_root
# If it's named differently, change 'app' to the correct filename without .py
try:
    from chatbot import load_chroma_vector_store, create_custom_prompt
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the project root and contains load_chroma_vector_store and create_custom_prompt.")
    sys.exit(1)

# Load environment and data
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.json')
try:
    with open(test_data_path, 'r', encoding="utf-8") as f:
        test_data = json.load(f)
except FileNotFoundError:
    print(f"Error: test_data.json not found at {test_data_path}")
    sys.exit(1)

# Initialize models for metrics
try:
    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bert_scorer_model = BERTScorer(lang="fr", model_type="bert-base-multilingual-cased") # Renamed to avoid conflict
except Exception as e:
    print(f"Error initializing metric models: {e}")
    sys.exit(1)

def normalize(text):
    """Normalize text by removing accents, lowercasing, and stripping whitespace."""
    if not isinstance(text, str): # Handle potential non-string inputs
        text = str(text)
    return unidecode(text.strip().lower())

def calculate_metrics(response, reference):
    """Calculate all metrics with normalization and sacreBLEU"""
    norm_response = normalize(response)
    norm_reference = normalize(reference)

    # BLEU with sacreBLEU
    # sacreBLEU expects a list of references
    bleu = sacrebleu.sentence_bleu(norm_response, [norm_reference], smooth_method='exp').score / 100.0

    # ROUGE
    rouge_scores = rouge.score(norm_reference, norm_response) # target, prediction

    # BERTScore
    # Ensure inputs are lists of strings
    P, R, F1 = bert_scorer_model.score([norm_response], [norm_reference])

    # Semantic similarity
    emb_ref = sentence_model.encode(norm_reference)
    emb_resp = sentence_model.encode(norm_response)
    # Ensure embeddings are 2D arrays for cosine_similarity
    cosine_sim = float(cosine_similarity(emb_ref.reshape(1, -1), emb_resp.reshape(1, -1))[0][0])

    return {
        "bleu_score": bleu,
        "rouge1": rouge_scores['rouge1'].fmeasure,
        "rouge2": rouge_scores['rouge2'].fmeasure,
        "rougeL": rouge_scores['rougeL'].fmeasure,
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item(),
        "cosine_similarity": cosine_sim,
        "response_raw": response, # Keep raw for inspection
        "reference_raw": reference,
        "response_normalized": norm_response,
        "reference_normalized": norm_reference
    }
    
def evaluate_model():
    """Main evaluation function"""
    # Initialize RAG system components
    chroma_db_path_env = os.getenv("CHROMADB_PATH", "vectorDB/chroma_db_cleaned") # Default from your app
    chroma_db_full_path = os.path.join(project_root, chroma_db_path_env)

    if not os.path.exists(chroma_db_full_path):
        print(f"Chroma DB path not found: {chroma_db_full_path}")
        print("Please ensure CHROMADB_PATH in .env is correct or the default path exists.")
        return
        
    vectordb = load_chroma_vector_store(chroma_db_full_path)
    
    # Configure LLM
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_NAME", "gemini-pro"), # Default from your app
        temperature=0.7,  # Use the temperature you found optimal or a default
        google_api_key=os.getenv("GEMINI_API_KEY"),
        max_output_tokens=2048
    )
    
    # Create ConversationBufferMemory (as used in your app)
    # For isolated Q&A, this memory will be cleared before each item
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question", # Aligns with the new prompt's expectation for the memory
        return_messages=False # Aligns with app.py for string history
    )
    
    # Create custom prompt (as used in your app)
    # This prompt expects 'question', 'context', and 'chat_history'
    prompt_template = create_custom_prompt()
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Standard chain type for RAG
        retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs={
            "prompt": prompt_template,
            "memory": memory # Pass the memory instance here
        },
        return_source_documents=True, # Good for debugging, not strictly needed for metrics
        input_key="query" # RetrievalQA itself expects 'query' as the top-level input
    )
    
    results = []
    print(f"Starting evaluation with {len(test_data)} test items...")
    for i, item in enumerate(test_data):
        print(f"Processing item {i+1}/{len(test_data)}: {item['question'][:50]}...")
        try:
            # IMPORTANT: Clear memory before each independent question
            memory.chat_memory.clear() 
            
            # The 'query' key here is what RetrievalQA chain expects as its input_key
            # It will internally pass item["question"] to the prompt as 'question' (due to memory's input_key)
            # and also use it for retrieval.
            response_data = qa_chain.invoke({"query": item["question"]})
            response_text = response_data['result']
            
            metrics = calculate_metrics(response_text, item["reference_answer"])
            metrics["question"] = item["question"]
            # Optionally include source documents for review
            # metrics["source_documents"] = [doc.page_content for doc in response_data.get('source_documents', [])]
            results.append(metrics)
            time.sleep(6) 

        except Exception as e:
            print(f"Error processing question '{item['question']}': {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Optionally, save error information
            results.append({
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            continue # Continue to the next item
    
    # Save results
    output_file_path = os.path.join(os.path.dirname(__file__), 'model_metrics_results.json')
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to {output_file_path}")
    
    # Filter out items with errors for averaging metrics
    valid_results = [r for r in results if "error" not in r]

    if valid_results:
        print("\nModel Evaluation Summary (based on valid results):")
        print(f"Total questions processed: {len(test_data)}")
        print(f"Valid responses for metrics: {len(valid_results)}")
        
        avg_bleu = np.mean([r['bleu_score'] for r in valid_results if 'bleu_score' in r])
        avg_rouge1 = np.mean([r['rouge1'] for r in valid_results if 'rouge1' in r])
        avg_rouge2 = np.mean([r['rouge2'] for r in valid_results if 'rouge2' in r])
        avg_rougeL = np.mean([r['rougeL'] for r in valid_results if 'rougeL' in r])
        avg_bert_f1 = np.mean([r['bert_f1'] for r in valid_results if 'bert_f1' in r])
        avg_cosine_sim = np.mean([r['cosine_similarity'] for r in valid_results if 'cosine_similarity' in r])

        print(f"Average BLEU: {avg_bleu:.4f}")
        print(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-2 F1: {avg_rouge2:.4f}")
        print(f"Average ROUGE-L F1: {avg_rougeL:.4f}")
        print(f"Average BERTScore F1: {avg_bert_f1:.4f}")
        print(f"Average Cosine Similarity: {avg_cosine_sim:.4f}")
    elif results and any("error" in r for r in results):
        print("\nModel Evaluation Summary: All items resulted in errors.")
    else:
        print("No valid results were generated to calculate average metrics.")

if __name__ == "__main__":
    evaluate_model()