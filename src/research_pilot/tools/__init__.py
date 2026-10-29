"""Tool registry: JSON schemas sent to the model, and a Toolbox that validates input and dispatches."""

import httpx

from ..config import Settings
from .base import ToolResult, error
from .cache import DiskCache
from .calculator import calculator
from .fetch import fetch_page
from .search import web_search

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


class Toolbox:
    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        self.settings = settings
        self.http = http or httpx.Client()
        self.cache = DiskCache(settings.cache_dir, enabled=settings.use_cache)

    def run(self, name: str, tool_input: dict) -> ToolResult:
        if name not in _REQUIRED:
            return error(f"unknown tool '{name}'")
        missing = [field for field in _REQUIRED[name] if not tool_input.get(field)]
        if missing:
            return error(f"missing required argument(s): {', '.join(missing)}")

        if name == "calculator":
            return calculator(str(tool_input["expression"]))

        cached = self.cache.get(name, tool_input)
        if cached is not None:
            return ToolResult(ok=True, content=cached, cached=True)

        if name == "web_search":
            result = web_search(self.http, self.settings.tavily_api_key, str(tool_input["query"]), tool_input.get("k", 5))
        else:
            result = fetch_page(self.http, str(tool_input["url"]))

        if result.ok:
            self.cache.set(name, tool_input, result.content)
        return result
