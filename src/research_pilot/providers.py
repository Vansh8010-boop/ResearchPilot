"""Model providers behind one interface.

Agents speak the Anthropic message format (content blocks, tool_use / tool_result). Each backend
takes a request in that format and returns a response object with the same shape as an Anthropic
Message: `.content` (blocks with .type text / tool_use), `.stop_reason`, `.usage`.

- AnthropicBackend: Claude via the official SDK.
- OpenAICompatBackend: any OpenAI-compatible chat endpoint. Used for Groq (free tier) and Ollama
  (local, no key). Translates messages and tools in both directions and retries rate limits.
"""

import json
import time
from types import SimpleNamespace

import anthropic
import httpx

from .config import Settings
from .tools import TOOL_SCHEMAS

ANTHROPIC_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
MAX_RETRY_WAIT_S = 90  # waits longer than this (e.g. a daily limit) are raised instead of slept through
MAX_RETRIES = 6


class ProviderError(Exception):
    pass


class AnthropicBackend:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.client = client or anthropic.Anthropic()

    def create(self, system: str, messages: list, tool_choice: str | None, json_schema: dict | None):
        output_config: dict = {"effort": self.settings.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs: dict = {}
        if tool_choice is not None:
            kwargs = {"tools": TOOL_SCHEMAS, "tool_choice": {"type": tool_choice}}
        if self.settings.model in ANTHROPIC_FALLBACK_MODELS:
            # On a safety refusal the API retries on a fallback model instead of returning nothing.
            kwargs |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            return self.client.beta.messages.create(
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                system=system,
                output_config=output_config,
                messages=messages,
                **kwargs,
            )
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc


# --- OpenAI-compatible -------------------------------------------------------------------------

_FINISH_TO_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}


def _block_get(block, key):
    return block[key] if isinstance(block, dict) else getattr(block, key)


def to_openai_messages(system: str, messages: list) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue

        if message["role"] == "assistant":
            text = "\n".join(_block_get(b, "text") for b in content if _block_get(b, "type") == "text")
            calls = [
                {"id": _block_get(b, "id"), "type": "function",
                 "function": {"name": _block_get(b, "name"), "arguments": json.dumps(_block_get(b, "input"))}}
                for b in content if _block_get(b, "type") == "tool_use"
            ]
            entry: dict = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue

        texts = []
        for block in content:
            if block["type"] == "tool_result":
                body = block["content"]
                out.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                            "content": f"ERROR: {body}" if block.get("is_error") else body})
            elif block["type"] == "text":
                texts.append(block["text"])
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_openai_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in TOOL_SCHEMAS]


def from_openai_response(data: dict) -> SimpleNamespace:
    choice = data["choices"][0]
    message = choice["message"]
    blocks = []
    if message.get("content"):
        blocks.append(SimpleNamespace(type="text", text=message["content"]))
    for call in message.get("tool_calls") or []:
        try:
            tool_input = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_input = {}  # the toolbox reports the missing arguments back to the model
        if not isinstance(tool_input, dict):
            tool_input = {}
        blocks.append(SimpleNamespace(type="tool_use", id=call["id"], name=call["function"]["name"], input=tool_input))

    if any(b.type == "tool_use" for b in blocks):
        stop_reason = "tool_use"
    else:
        stop_reason = _FINISH_TO_STOP.get(choice.get("finish_reason"), "end_turn")
    usage = data.get("usage") or {}
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens")),
    )


class OpenAICompatBackend:
    def __init__(self, settings: Settings, http: httpx.Client | None = None, sleep=time.sleep):
        if not settings.base_url:
            raise ProviderError(f"no base URL configured for provider '{settings.provider}'")
        self.settings = settings
        self.http = http or httpx.Client()
        self.sleep = sleep

    def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.settings.api_key}"} if self.settings.api_key else {}
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.http.post(url, json=payload, headers=headers, timeout=180.0)
            except httpx.HTTPError as exc:
                if attempt == MAX_RETRIES:
                    raise ProviderError(f"request to {url} failed: {type(exc).__name__}: {exc}") from exc
                self.sleep(min(2 ** attempt, 30))
                continue

            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 or response.status_code >= 500:
                wait = float(response.headers.get("retry-after", 2 ** attempt))
                if wait > MAX_RETRY_WAIT_S or attempt == MAX_RETRIES:
                    raise ProviderError(f"HTTP {response.status_code} (retry after {wait:.0f}s): {response.text[:300]}")
                self.sleep(wait)
                continue
            raise ProviderError(f"HTTP {response.status_code}: {response.text[:300]}")
        raise ProviderError("retries exhausted")

    def create(self, system: str, messages: list, tool_choice: str | None, json_schema: dict | None):
        if json_schema is not None:
            # Plain JSON mode is supported by both Groq and Ollama; the schema goes in the prompt.
            system += ("\n\nRespond with only a JSON object (no prose, no code fences) matching this schema:\n"
                       + json.dumps(json_schema))
        payload: dict = {
            "model": self.settings.model,
            "messages": to_openai_messages(system, messages),
            "max_tokens": min(self.settings.max_tokens, 8192),
        }
        if tool_choice is not None:
            payload |= {"tools": to_openai_tools(), "tool_choice": tool_choice}
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        return from_openai_response(self._post(payload))


def make_backend(settings: Settings, client: anthropic.Anthropic | None = None):
    if settings.provider == "anthropic":
        return AnthropicBackend(settings, client)
    return OpenAICompatBackend(settings)
