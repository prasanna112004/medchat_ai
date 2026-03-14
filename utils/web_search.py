from duckduckgo_search import DDGS


def web_search(query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)

        if not results:
            return "No web results found."

        formatted = []
        for r in results:
            formatted.append(
                f"Source: {r['href']}\n"
                f"Title: {r['title']}\n"
                f"Summary: {r['body']}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception:
        return "Web search unavailable. Please try again."
