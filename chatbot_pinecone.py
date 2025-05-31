from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Initialize environment
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("MODEL_NAME", "gemini-pro")  # Default model

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "cynoia-vector-db")

# Streamlit UI setup
st.set_page_config(page_title="Assistant Cynoia", page_icon="🌍")
st.title("🌍 Assistant Virtuel Cynoia")
st.caption("Posez vos questions sur la plateforme SaaS panafricaine")

def get_pinecone_retriever():
    """Initialize Pinecone vector store using the dedicated package"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Using the dedicated PineconeVectorStore
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")  # Pass API key directly
    )
    return vector_store.as_retriever(search_kwargs={"k": 10})

def create_qa_chain():
    prompt_template = """Tu es l'assistant officiel de Cynoia, une plateforme SaaS panafricaine aidant les entreprises africaines à mieux collaborer. 
Réponds toujours en français, avec un ton professionnel, chaleureux et clair.

Consignes importantes :
- Base-toi principalement sur le contexte fourni.
- Si une information n'est pas explicitement dans le contexte, essaie de répondre de manière logique et utile en te basant sur ta connaissance générale.
- Ne dis que tu ne sais pas que si la question est vraiment hors sujet.
- Encourage l'utilisateur à visiter le site officiel pour plus de détails.
- Encourage les utilisateurs à découvrir et adopter Cynoia.
- la reponse doit être concise et pertinente.

Voici les informations disponibles (contexte) :
{context}

Question : {question}

Réponse :"""
    
    return RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_key,
            temperature=0.7,
            convert_system_message_to_human=True
        ),
        chain_type="stuff",
        retriever=get_pinecone_retriever(),
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        },
        return_source_documents=True
    )

# Initialize session state
if "qa_chain" not in st.session_state:
    with st.spinner("Chargement des connaissances Cynoia..."):
        try:
            st.session_state.qa_chain = create_qa_chain()
        except Exception as e:
            st.error(f"Échec de l'initialisation: {str(e)}")
            st.stop()

# Rest of your Streamlit chat interface remains the same...

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider sur Cynoia aujourd'hui ?"}
    ]

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans nos ressources..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": prompt})
                response = result['result']
                
                st.markdown(response)
                with st.expander("Sources utilisées"):
                    for doc in result['source_documents']:
                        st.caption(f"📌 {doc.metadata.get('source', 'Source inconnue')}")
                        st.text(doc.page_content[:200] + "...")

            except Exception as e:
                response = f"⚠️ Désolé, une erreur est survenue: {str(e)}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})