def clean_sitemap_file(input_file="sitemap.txt", output_file="sitemap_clean.txt", overwrite=False):
    with open(input_file, "r", encoding="utf-8") as f:
        urls = f.read().split()

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    final_output = input_file if overwrite else output_file

    with open(final_output, "w", encoding="utf-8") as f:
        for url in unique_urls:
            f.write(url + "\n")

    print(f"✅ Cleaned sitemap: {len(urls)} → {len(unique_urls)} unique URLs")
    print(f"📁 Saved to: {final_output}")


# Example usage:
if __name__ == "__main__":
    clean_sitemap_file(overwrite=False)  # set to False to save as sitemap_clean.txt
