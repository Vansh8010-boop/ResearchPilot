"""Tool registry: JSON schemas sent to the model, and a Toolbox that validates input and dispatches."""

import json
import socket
import time
from functools import partial
from urllib.parse import urlparse

import httpx

from ..config import Settings
from .base import ToolResult, error
from .cache import DiskCache
from .calculator import calculator
from .fetch import fetch_page
from .search import web_search
from .security import Provenance, private_address_reason, wrap_untrusted

TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": (
            "Search the web. Returns a JSON list of results with title, url and a short snippet. "
            "Snippets are often incomplete; use fetch_page on a promising url to read the full page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, a few keywords work best."},
                "k": {"type": "integer", "description": "Number of results, 1-10. Default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": (
            "Download a web page and return its main text (truncated to about 8000 characters). "
            "Use it to read a page found with web_search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full http(s) URL."}},
            "required": ["url"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression exactly. Supports + - * / // % **, parentheses, "
            "sqrt, log, log10, exp, abs, round, min, max, pi and e. Write numbers without commas. "
            "Always use this for arithmetic instead of computing in your head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. (125.7e6 / 695662) * 2"}},
            "required": ["expression"],
        },
    },
]

_REQUIRED = {s["name"]: s["input_schema"]["required"] for s in TOOL_SCHEMAS}


RETRY_DELAYS_S = (1.0, 3.0)


class Toolbox:
    def __init__(self, settings: Settings, http: httpx.Client | None = None, sleep=time.sleep,
                 resolve=socket.getaddrinfo):
        self.settings = settings
        self.http = http or httpx.Client()
        self.cache = DiskCache(settings.cache_dir, enabled=settings.use_cache)
        self.sleep = sleep
        self.resolve = resolve
        self.provenance = Provenance()

    def start_run(self, question: str) -> None:
        """Reset per-run state. URLs in the question count as trusted provenance."""
        self.provenance = Provenance.from_text(question)

    def run(self, name: str, tool_input: dict) -> ToolResult:
        if name not in _REQUIRED:
            return error(f"unknown tool '{name}'")
        missing = [field for field in _REQUIRED[name] if not tool_input.get(field)]
        if missing:
            return error(f"missing required argument(s): {', '.join(missing)}")

        if name == "calculator":
            return calculator(str(tool_input["expression"]))

        defenses = self.settings.defenses
        if defenses and name == "fetch_page":
            reason = self.provenance.check(str(tool_input["url"]))
            if reason:
                return ToolResult(ok=False, content=reason, blocked=True)

        result = self._run_web_tool(name, tool_input, retries=RETRY_DELAYS_S if defenses else ())
        if not result.ok:
            return result

        if name == "web_search":
            # Only the result URLs themselves, never URLs mentioned inside snippets.
            for item in json.loads(result.content):
                self.provenance.add_url(item["url"])
        else:
            self.provenance.add_url(str(tool_input["url"]))
        if defenses:
            source = tool_input["url"] if name == "fetch_page" else f"search: {tool_input['query']}"
            result.content = wrap_untrusted(name, str(source), result.content)
        return result

    def _run_web_tool(self, name: str, tool_input: dict, retries: tuple[float, ...]) -> ToolResult:
        cached = self.cache.get(name, tool_input)
        if cached is not None:
            return ToolResult(ok=True, content=cached, cached=True)

        for delay in (*retries, None):
            if name == "web_search":
                result = web_search(self.http, self.settings.tavily_api_key, str(tool_input["query"]),
                                    tool_input.get("k", 5))
            else:
                guard = (partial(private_address_reason, resolve=self.resolve)
                         if self.settings.block_private_urls else None)
                result = fetch_page(self.http, str(tool_input["url"]), guard)
            if result.ok or not result.retryable or delay is None:
                break
            self.sleep(delay)

        # Local pages (benchmark fixtures) change while we edit them, so never cache them.
        is_local = name == "fetch_page" and urlparse(str(tool_input["url"])).hostname in ("127.0.0.1", "localhost")
        if result.ok and not is_local:
            self.cache.set(name, tool_input, result.content)
        return result
