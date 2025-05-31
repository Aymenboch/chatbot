import os
import shutil
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import re

load_dotenv()

# 💡 Optional: Clean repeated UI noise (if not done during scraping)
def clean_text(raw_text: str) -> str:
    lines = raw_text.splitlines()
    seen = set()
    clean_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.lower() in seen:
            continue
        if re.search(r"(login|sign up|create your free workspace|book a demo)", line, re.IGNORECASE):
            continue
        if len(line) < 10:
            continue
        seen.add(line.lower())
        clean_lines.append(line)

    return "\n".join(clean_lines)

# 💡 Load JSON and convert to LangChain documents with chunking
def load_json_and_create_chunks(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for page in data:
        url = page.get("url", "")
        title = page.get("title", "")
        section = page.get("section", "")
        content = clean_text(page.get("content", ""))

        if not content.strip():
            continue

        chunks = splitter.split_text(content)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": url,
                    "title": title,
                    "section": section
                }
            )
            docs.append(doc)
    
    return docs

# 💡 Create and store in vector DB
def create_vector_store(documents, persist_directory: str):
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print("Creating new vector store...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

# 💡 Main entry point
def main():
    json_path = "C:/Users/aayme/Desktop/chatbot/web_scraping/cynoia_scraped_cleaned.json"
    db_dir = "C:/Users/aayme/Desktop/chatbot/vectorDB/chroma_db_cleaned"

    print("Processing JSON and creating document chunks...")
    docs = load_json_and_create_chunks(json_path)
    print(f"Generated {len(docs)} document chunks")

    print("Storing in Chroma vector DB...")
    create_vector_store(docs, db_dir)
    print(f"✅ Vector store ready at {db_dir}")

if __name__ == "__main__":
    main()
