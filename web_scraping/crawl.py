from firecrawl import FirecrawlApp, ScrapeOptions

app = FirecrawlApp(api_key="fc-e38059ae1e7b48c2961849537bc25b57")

# Crawl a website:
crawl_result = app.crawl_url(
  'https://cynoia.com/', 
  limit=10, 
  scrape_options=ScrapeOptions(formats=['markdown', 'html']),
)
print(crawl_result)