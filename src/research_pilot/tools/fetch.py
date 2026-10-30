"""fetch_page tool: download a URL and return its main readable text."""

from typing import Callable
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura

from .base import ToolResult, error

MAX_CHARS = 8000
MAX_REDIRECTS = 5
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


def fetch_page(http: httpx.Client, url: str, guard: Callable[[str], str | None] | None = None) -> ToolResult:
    """`guard(url)` returns a reason to refuse a URL (e.g. it points at a private address). With a
    guard, redirects are followed one hop at a time so every hop is checked, not just the first."""
    if urlparse(url).scheme not in ("http", "https"):
        return error(f"only http(s) URLs are supported, got '{url}'")

    try:
        if guard is None:
            response = http.get(url, headers={"User-Agent": USER_AGENT}, timeout=15.0, follow_redirects=True)
        else:
            current = url
            for _ in range(MAX_REDIRECTS + 1):
                reason = guard(current)
                if reason:
                    return ToolResult(ok=False, content=reason, blocked=True)
                response = http.get(current, headers={"User-Agent": USER_AGENT}, timeout=15.0, follow_redirects=False)
                if not (response.is_redirect and "location" in response.headers):
                    break
                current = urljoin(current, response.headers["location"])
            else:
                return error(f"too many redirects from {url}")
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
