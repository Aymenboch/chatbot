import os
import json
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec  # Now imports correctly!

load_dotenv()

# --- Config ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "cynoia-vector-db"

# --- Step 1: Clean text ---
def clean_text(raw_text: str) -> str:
    lines = raw_text.splitlines()
    seen = set()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in seen and not re.match(
            r"(login|sign up|book a demo|create your free workspace)", line.lower()
        ):
            seen.add(line)
            clean_lines.append(line)
    return "\n".join(clean_lines)

# --- Step 2: Load JSON and chunk documents ---
def load_json_and_create_chunks(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []

    for page in data:
        content = clean_text(page["content"])
        text_chunks = splitter.split_text(content)

        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{page.get('id', page.get('url', 'unknown'))}_chunk_{i}"
            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": page.get("url", "unknown"),
                    "title": page.get("title", "unknown")
                }
            })
    return chunks

# --- Step 3: Initialize Pinecone index ---
def init_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # 'all-MiniLM-L6-v2' embedding size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    return pc.Index(INDEX_NAME)

# --- Step 4: Embed and upsert chunks ---
def embed_and_upsert(chunks, index, model):
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [item["text"] for item in batch]
        embeddings = model.encode(texts).tolist()

        vectors = []
        for j, emb in enumerate(embeddings):
            meta = batch[j]["metadata"]
            vectors.append({
                "id": batch[j]["id"],
                "values": emb,
                "metadata": meta
            })

        index.upsert(vectors=vectors)
        print(f"Upserted batch {i//batch_size + 1}")

# --- Step 5: Main ---
def main():
    json_path = "C:/Users/aayme/Desktop/chatbot/web_scraping/cynoia_scraped.json"
    print("Loading and chunking JSON...")
    chunks = load_json_and_create_chunks(json_path)
    print(f"Created {len(chunks)} chunks")

    print("Initializing Pinecone index...")
    index = init_pinecone_index()

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding and upserting vectors...")
    embed_and_upsert(chunks, index, model)

    print(f"✅ Pinecone vector DB '{INDEX_NAME}' is ready!")

if __name__ == "__main__":
    main()