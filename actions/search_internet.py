from duckduckgo_search import DDGS

def search_internet(query: str) -> str:
    print(f"[SEARCH] Searching for: {query}")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            print(f"[SEARCH] result title: {results[0]['title'] if results else 'No results found'}, total searches: {len(results) if results else 0}")
            if not results:
                return "No results found."
            return "\n\n".join([f"{r['title']}\n{r['body']}\n{r['href']}" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"
