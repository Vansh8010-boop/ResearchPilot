"""fetch_page tool: download a URL and return its main readable text."""

from urllib.parse import urlparse

import httpx
import trafilatura

from .base import ToolResult, error

MAX_CHARS = 8000
MAX_BYTES = 3_000_000
TEXT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
USER_AGENT = "Mozilla/5.0 (compatible; ResearchPilot/0.1; +https://github.com/)"


def extract_text(body: str, content_type: str) -> str | None:
    if content_type.startswith("text/plain"):
        return body.strip() or None
    return trafilatura.extract(body, include_comments=False, include_tables=True)


def truncate(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[truncated: showing {limit} of {len(text)} characters]"


def fetch_page(http: httpx.Client, url: str) -> ToolResult:
    if urlparse(url).scheme not in ("http", "https"):
        return error(f"only http(s) URLs are supported, got '{url}'")

    try:
        response = http.get(url, headers={"User-Agent": USER_AGENT}, timeout=15.0, follow_redirects=True)
    except httpx.TimeoutException:
        return error(f"fetch timed out: {url}; try another source", retryable=True)
    except httpx.HTTPError as exc:
        return error(f"fetch failed: {type(exc).__name__}: {exc}", retryable=True)
    if response.status_code != 200:
        return error(f"fetch failed: HTTP {response.status_code} for {url}; try another source",
                     retryable=response.status_code == 429 or response.status_code >= 500)

    content_type = response.headers.get("content-type", "").lower()
    if not content_type.startswith(TEXT_TYPES):
        return error(f"unsupported content type '{content_type}' at {url}")
    if len(response.content) > MAX_BYTES:
        return error(f"page too large ({len(response.content)} bytes) at {url}")

    text = extract_text(response.text, content_type)
    if not text:
        return error(f"no readable text found at {url}")
    return ToolResult(ok=True, content=truncate(text))
