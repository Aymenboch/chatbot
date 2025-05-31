import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
import json
import time
import re
import nltk
from nltk.tokenize import sent_tokenize
 # for sentence tokenization
nltk.download('punkt_tab')

def extract_semantic_chunks(html, url, title):
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    content = []
    section_title = ""
    buffer = []

    def flush_buffer():
        if buffer:
            text_block = " ".join(buffer).strip()
            if text_block:
                # Split long blocks into smaller ones using sentence tokenizer
                sentences = sent_tokenize(text_block)
                for i in range(0, len(sentences), 5):  # ~5 sentences per chunk
                    chunk = " ".join(sentences[i:i+5])
                    content.append({
                        "url": url,
                        "title": title,
                        "section": section_title,
                        "content": chunk
                    })
            buffer.clear()

    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
        if isinstance(tag, Tag):
            if tag.name in ['h1', 'h2', 'h3']:
                flush_buffer()
                section_title = tag.get_text(strip=True)
            elif tag.name in ['p', 'li']:
                text = tag.get_text(strip=True)
                if text:
                    buffer.append(text)

    flush_buffer()
    return content

def chunk_text(text, max_length=500, overlap=100):
    """
    Split text into chunks of max_length words with some overlap.
    """
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i+max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap  # slide window forward

    return chunks

def load_urls_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        urls = content.split()
    return urls

def extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Try to get <main> content if exists
    main = soup.find("main")
    if main:
        text = main.get_text(separator="\n", strip=True)
        if text:
            return text

    # Fallback: get all visible text excluding script/style
    for script in soup(["script", "style", "noscript"]):
        script.extract()

    text = soup.get_text(separator="\n", strip=True)

    # Optionally: keep only paragraphs
    paragraphs = soup.find_all("p")
    if paragraphs:
        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    
    return text

def scrape_pages(urls):
    items = []
    headers = {
        "User-Agent": "CynoiaScraperBot/1.0"
    }

    for i, url in enumerate(urls, 1):
        try:
            print(f"[{i}/{len(urls)}] Fetching {url}")
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, "html.parser")

            title = soup.title.string.strip() if soup.title else ""
            chunks = extract_semantic_chunks(resp.text, url, title)

            if chunks:
                items.extend(chunks)
            else:
                print(f"Warning: no semantic chunks extracted from {url}")

            time.sleep(1)  # polite delay

        except Exception as e:
            print(f"Error fetching {url}: {e}")

    return items

def main():
    sitemap_file = "sitemap_clean.txt"
    urls = load_urls_from_file(sitemap_file)

    print(f"Loaded {len(urls)} URLs from {sitemap_file}")

    scraped_data = scrape_pages(urls)

    if scraped_data:

        with open("cynoia_scraped.json", "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)

        print(f"Scraped {len(scraped_data)} pages. Saved to cynoia_scraped.csv and cynoia_scraped.json")

    else:
        print("No data scraped.")

if __name__ == "__main__":
    main()
