from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
from dotenv import load_dotenv
import sqlite3
import hashlib
import base64
from datetime import datetime

# Load environment variables
load_dotenv()

# Original functions (unchanged)
def load_chroma_vector_store(db_dir):
    """Load the Chroma vector store from the persisted directory."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return Chroma(persist_directory=db_dir, embedding_function=embeddings)

def create_custom_prompt():
    template2 = """Tu es l'assistant officiel de Cynoia, une plateforme SaaS panafricaine qui aide les entreprises africaines à mieux collaborer.
Réponds toujours en français, avec un ton professionnel, chaleureux et clair.

Directives à suivre :
- Appuie-toi principalement sur le contexte fourni.
- Si une information n’est pas explicitement dans le contexte, tu peux répondre de manière logique et utile, en te basant sur ta connaissance générale.
- Ne dis que tu ne sais pas que si la question est vraiment hors sujet.
- Ne mentionne jamais explicitement que tu utilises un "contexte".
- Ne fournis jamais d'informations inventées.
- Termine si possible par une phrase encourageant à découvrir ou adopter Cynoia.
- Invite l'utilisateur à consulter le site officiel pour des détails supplémentaires.
- Reste toujours professionnel, bienveillant et synthétique.

Informations disponibles :
{context}

Question : {question}

Réponse :
"""

    template = """Tu es l'assistant officiel de Cynoia.

Consignes :
- Donne une réponse directe, sans phrases d’introduction.
- Sois concis, factuel et reste dans le contexte.
- Ne commente pas le contexte ni ne mentionne que tu l’utilises.
- Ne répète pas la question.
- Évite les formules promotionnelles ou génériques.

Historique de la conversation :
{chat_history}

Informations disponibles :
{context}

Question : {question}

Réponse :"""

    template3 = """Tu es l'assistant officiel de Cynoia, une plateforme SaaS panafricaine qui aide les entreprises africaines à mieux collaborer.
Réponds toujours selon la langue de l'utilisateur, avec un ton professionnel, chaleureux et clair.

Directives à suivre :
- Appuie-toi principalement sur le contexte fourni.
- Si une information n'est pas explicitement dans le contexte, tu peux répondre de manière logique et utile.
- Ne dis que tu ne sais pas que si la question est vraiment hors sujet.
- Ne mentionne jamais explicitement que tu utilises un "contexte".
- Ne fournis jamais d'informations inventées.
- Termine si possible par une phrase encourageant à découvrir ou adopter Cynoia.
- Invite l'utilisateur à consulter le site officiel pour des détails supplémentaires.
- Reste toujours professionnel, bienveillant et synthétique.

Historique de la conversation :
{chat_history}

Informations disponibles :
{context}

Question : {question}

Réponse :"""

    return PromptTemplate(template=template3, input_variables=["chat_history", "context", "question"])



def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_base64 = get_image_base64("C:/Users/aayme/Desktop/chatbot/cynoia.jpg")


# Database setup
def init_db():
    conn = sqlite3.connect('cynoia_chat.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  session_id TEXT,
                  prompt TEXT,
                  response TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    conn.commit()
    conn.close()

init_db()

# Authentication functions
def create_user(username, password):
    conn = sqlite3.connect('cynoia_chat.db')
    c = conn.cursor()
    try:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                 (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError: # Handles UNIQUE constraint violation for username
        return False
    except Exception: # Catch other potential DB errors
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('cynoia_chat.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", 
              (username, hashed_pw))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def save_chat(user_id, session_id, prompt, response):
    conn = sqlite3.connect('cynoia_chat.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_id, session_id, prompt, response) VALUES (?, ?, ?, ?)",
              (user_id, session_id, prompt, response))
    conn.commit()
    conn.close()

def get_chat_history(user_id, session_id=None):
    conn = sqlite3.connect('cynoia_chat.db')
    c = conn.cursor()
    
    if session_id:
        c.execute("""
            SELECT prompt, response, timestamp 
            FROM chats 
            WHERE user_id=? AND session_id=? 
            ORDER BY timestamp
            """, (user_id, session_id))
    else:
        # Get distinct sessions with their latest timestamp and first prompt for better display
        c.execute("""
            SELECT 
                session_id, 
                MAX(timestamp) as last_used,
                (SELECT prompt FROM chats c2 WHERE c2.session_id = chats.session_id AND c2.user_id = chats.user_id ORDER BY timestamp ASC LIMIT 1) as first_prompt
            FROM chats
            WHERE user_id=?
            GROUP BY session_id
            ORDER BY last_used DESC
            """, (user_id,))
    
    result = c.fetchall()
    conn.close()
    return result

# Streamlit UI setup
st.set_page_config(page_title="Assistant Cynoia", page_icon="C:/Users/aayme/Desktop/chatbot/cynoia.jpg", layout='centered', initial_sidebar_state='expanded')

# Initialize session state keys if they don't exist
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Authentication form / Main App Logic
if not st.session_state.authenticated:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/jpg;base64,{image_base64}" alt="logo" width="40" style="margin-right: 10px;"/>
            <h1 style="margin: 0;">Connexion à Cynoia</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    tab1, tab2 = st.tabs(["Connexion", "Inscription"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter")
            if submitted:
                user_id = verify_user(username, password)
                if user_id:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    # Initialize for a new session upon login
                    st.session_state.current_session = str(datetime.now().timestamp())
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Bonjour ! Je suis l'assistant Cynoia. Posez-moi vos questions sur notre plateforme.", "avatar":'C:/Users/aayme/Desktop/chatbot/cynoia.jpg'}
                    ]
                    # Clear any old memory or QA chain to ensure fresh start
                    st.session_state.pop('memory', None)
                    st.session_state.pop('qa_chain', None)
                    st.rerun()
                else:
                    st.error("Identifiants incorrects")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choisissez un nom d'utilisateur")
            new_password = st.text_input("Choisissez un mot de passe", type="password")
            confirm_password = st.text_input("Confirmez le mot de passe", type="password")
            submitted = st.form_submit_button("S'inscrire")
            if submitted:
                if not new_username or not new_password:
                    st.error("Le nom d'utilisateur et le mot de passe ne peuvent pas être vides.")
                elif new_password != confirm_password:
                    st.error("Les mots de passe ne correspondent pas")
                elif create_user(new_username, new_password):
                    st.success("Compte créé avec succès! Veuillez vous connecter.")
                else:
                    st.error("Ce nom d'utilisateur existe déjà ou une erreur est survenue.")
    st.stop() # Stop execution if not authenticated

# --- Authenticated App Starts Here ---
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/jpg;base64,{image_base64}" alt="logo" width="40" style="margin-right: 10px;"/>
            <h1 style="margin: 0;">Assistant Cynoia</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"Connecté en tant que: {st.session_state.username}")
with col2:
    if st.button("Se déconnecter", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        
        # Clear all session-specific data to ensure a clean slate
        keys_to_clear = ['messages', 'memory', 'qa_chain', 'current_session']
        for key in keys_to_clear:
            if key == 'memory' and key in st.session_state and hasattr(st.session_state.memory, 'chat_memory'):
                 st.session_state.memory.chat_memory.clear() # Specifically clear Langchain memory
            st.session_state.pop(key, None)
        
        st.rerun()

st.caption("Posez vos questions sur la plateforme SaaS panafricaine")

# Ensure current_session is initialized if it was cleared or never set (should be set on login)
if 'current_session' not in st.session_state:
    st.session_state.current_session = str(datetime.now().timestamp())

# Initialize RAG system and memory
if "qa_chain" not in st.session_state or "memory" not in st.session_state:
    with st.spinner("Chargement du système Cynoia..."):
        db_dir_env = os.getenv("CHROMADB_PATH", "vectorDB/chroma_db_cleaned") # Example: use env var or default
        db_dir = os.path.expanduser(db_dir_env) # Handles ~ in paths
        
        if not os.path.exists(db_dir):
            st.error(f"Le répertoire de la base de données vectorielle n'existe pas : {db_dir}")
            st.stop()
        
        vectordb = load_chroma_vector_store(db_dir)
        
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7,
            max_output_tokens=2048
        )
        
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question", 
            return_messages=False 
        )
        
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={
                "prompt": create_custom_prompt(),
                "memory": st.session_state.memory
            },
            return_source_documents=True,
            input_key="query"
        )

# Session management dropdown
col1, col2 = st.columns([3, 1])
with col2:
    previous_sessions_data = get_chat_history(st.session_state.user_id)
    session_options_map = {"✨ Nouvelle session ✨": "new_session_identifier"}

    if previous_sessions_data:
        for session_id, last_used, first_prompt_text in previous_sessions_data:
            try:
                dt_object = datetime.strptime(last_used.split(".")[0], '%Y-%m-%d %H:%M:%S')
                label = f"🗓️ {dt_object.strftime('%d/%m %H:%M')} - {first_prompt_text[:35]}..." if first_prompt_text else f"🗓️ {dt_object.strftime('%d/%m/%Y %H:%M')} - Session vide"
                session_options_map[label] = session_id
            except ValueError:
                session_options_map[f"Session ID: {session_id[:8]}..."] = session_id

    selected_session_label = st.selectbox(
        "Gérer les sessions:",
        list(session_options_map.keys()),
        key="session_selector",
        label_visibility="collapsed"
    )

with col1:
    st.write("Chat en cours")

selected_session_key = session_options_map[selected_session_label]
session_action_triggered = False

if selected_session_key == "new_session_identifier":
    # Check if the current session is already pristine or if we need to reset for a new one
    is_current_pristine = (
        st.session_state.current_session not in [s[0] for s in previous_sessions_data if s[0]] and
        len(st.session_state.get("messages", [])) <= 1 and # Allows for initial assistant message
        (not st.session_state.get("messages") or st.session_state.messages[0]["role"] == "assistant")
    )
    if not is_current_pristine:
        st.session_state.current_session = str(datetime.now().timestamp())
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Prêt pour une nouvelle conversation."}]
        if "memory" in st.session_state: st.session_state.memory.chat_memory.clear()
        session_action_triggered = True

elif selected_session_key != st.session_state.current_session:
    st.session_state.current_session = selected_session_key
    st.session_state.messages = [] 
    if "memory" in st.session_state: st.session_state.memory.chat_memory.clear()
    
    history_for_selected_session = get_chat_history(st.session_state.user_id, st.session_state.current_session)
    for prompt_text, response_text, _ in history_for_selected_session:
        st.session_state.messages.append({"role": "user", "content": prompt_text, "avatar":'C:/Users/aayme/Desktop/chatbot/cynoia.jpg'})
        st.session_state.messages.append({"role": "assistant", "content": response_text, "avatar":'C:/Users/aayme/Desktop/chatbot/cynoia.jpg'})
        if "memory" in st.session_state:
            st.session_state.memory.chat_memory.add_user_message(prompt_text)
            st.session_state.memory.chat_memory.add_ai_message(response_text)
    session_action_triggered = True

if session_action_triggered:
    st.rerun()


# Initialize UI chat history if it's somehow missing (e.g., after certain session switches)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis l'assistant Cynoia. Posez-moi vos questions sur notre plateforme.", "avatar" :'C:/Users/aayme/Desktop/chatbot/cynoia.jpg'}
    ]
elif not st.session_state.messages: # If messages list is empty after session load (e.g. empty session)
     st.session_state.messages = [
        {"role": "assistant", "content": "Cette session est vide. Posez votre première question!", "avatar":'C:/Users/aayme/Desktop/chatbot/cynoia.jpg'}
    ]


# Display chat messages
for message in st.session_state.messages:
    avatar = message.get("avatar") if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat interaction
if prompt := st.chat_input("Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar='C:/Users/aayme/Desktop/chatbot/cynoia.jpg'):
        with st.spinner("En train d'écrire..."):
            try:
                result = st.session_state.qa_chain({"query": prompt})
                response = result['result']
                
                st.markdown(response)
                #if result.get('source_documents'):
                #    with st.expander("Voir les sources"):
                #        for doc in result['source_documents']:
                #            st.caption(f"📄 {doc.metadata.get('source', 'Source inconnue')}")
                #            st.code(doc.page_content[:200] + "...", language="markdown")
                
                save_chat(
                    st.session_state.user_id,
                    st.session_state.current_session,
                    prompt,
                    response
                )
            
            except Exception as e:
                response = f"Désolé, une erreur est survenue: {str(e)}"
                st.error(response)
                import traceback
                st.error(f"Trace:\n{traceback.format_exc()}") # More detailed error for debugging
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    # No need to st.rerun() here, chat_input handles updates for new messages.