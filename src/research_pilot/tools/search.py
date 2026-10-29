"""web_search tool backed by the Tavily search API."""

import json

import httpx

from .base import ToolResult, error

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 10


def web_search(http: httpx.Client, api_key: str | None, query: str, k: int = 5) -> ToolResult:
    if not api_key:
        return error("TAVILY_API_KEY is not set")
    query = query.strip()
    if not query:
        return error("empty query")
    try:
        k = max(1, min(int(k), MAX_RESULTS))
    except (TypeError, ValueError):
        return error(f"k must be an integer, got {k!r}")

    try:
        response = http.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": k, "search_depth": "basic"},
            timeout=20.0,
        )
    except httpx.HTTPError as exc:
        return error(f"search request failed: {type(exc).__name__}: {exc}", retryable=True)
    if response.status_code != 200:
        return error(f"search failed: HTTP {response.status_code}: {response.text[:200]}",
                     retryable=response.status_code == 429 or response.status_code >= 500)

    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:500]}
        for r in response.json().get("results", [])
    ]
    if not results:
        return error(f"no results for '{query}'; try a broader or rephrased query")
    return ToolResult(ok=True, content=json.dumps(results, ensure_ascii=False, indent=1))
