import json
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def remove_similar_sections(data, threshold=0.90):
    cleaned_data = []
    grouped_by_url = {}

    # Group entries by URL
    for entry in data:
        url = entry['url']
        grouped_by_url.setdefault(url, []).append(entry)

    # For each group, compare sections
    for url, entries in grouped_by_url.items():
        seen_sections = []
        for entry in entries:
            section = entry['section'].strip().lower()
            is_similar = False
            for seen in seen_sections:
                if similar(section, seen) >= threshold:
                    is_similar = True
                    break
            if not is_similar:
                seen_sections.append(section)
                cleaned_data.append(entry)

    return cleaned_data

# Load your original JSON
with open("cynoia_scraped.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Clean similar sections
cleaned_data = remove_similar_sections(data)

# Save the cleaned JSON
with open("cynoia_scraped_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Reduced from {len(data)} to {len(cleaned_data)} entries.")
