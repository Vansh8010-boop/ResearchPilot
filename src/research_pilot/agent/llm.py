"""Shared pieces for every agent: the model call (with tracing) and a bounded tool-use loop."""

import json
import time
from dataclasses import dataclass

from ..providers import ProviderError, ToolCallRejected
from ..tools import TOOL_SCHEMAS, Toolbox
from ..tools.base import ToolResult
from ..tracing import Trace
from .base import EventCallback


TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]
DUPLICATE_NOTE = ("Duplicate call blocked: you already made this exact call. Use its earlier result, "
                  "or try a different query or source.")


class LLMError(Exception):
    pass


def text_of(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


class LLM:
    """Wraps a provider backend: counts calls and tokens, records every call in the trace."""

    def __init__(self, backend, trace: Trace, on_event: EventCallback, token_budget: int | None = None):
        self.backend = backend
        self.trace = trace
        self.on_event = on_event
        self.token_budget = token_budget
        self.calls = 0
        self.tokens = 0

    def over_budget(self) -> bool:
        return self.token_budget is not None and self.tokens >= self.token_budget

    def call(self, phase: str, system: str, messages: list, tool_choice: str | None = "auto",
             json_schema: dict | None = None):
        """tool_choice: "auto", "none" (tools defined but not callable) or None (no tools at all)."""
        self.calls += 1
        started = time.perf_counter()
        try:
            response = self.backend.create(system, messages, tool_choice, json_schema)
        except ToolCallRejected as exc:
            self.trace.llm_call(phase, self.calls, time.perf_counter() - started, "tool_call_rejected", None, str(exc))
            raise
        except ProviderError as exc:
            self.trace.llm_call(phase, self.calls, time.perf_counter() - started, "api_error", None, str(exc))
            raise LLMError(str(exc)) from exc

        self.tokens += (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
        text = text_of(response)
        self.trace.llm_call(phase, self.calls, time.perf_counter() - started, response.stop_reason, response.usage, text)
        self.on_event("llm", {"phase": phase, "step": self.calls, "text": text, "stop_reason": response.stop_reason})
        return response


def _append_note(messages: list, note: str) -> None:
    """Add a text note to the last (user) message instead of starting a second user turn."""
    content = messages[-1]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    messages[-1]["content"] = [*content, {"type": "text", "text": note}]


@dataclass
class LoopResult:
    text: str
    stop_reason: str  # answered | max_steps | refusal | max_tokens
    turns: int


def tool_loop(llm: LLM, toolbox: Toolbox, phase: str, system: str, messages: list,
              max_turns: int, limit_note: str, dedupe: bool = False) -> LoopResult:
    """Call the model, run the tools it asks for, feed results back; at most `max_turns` tool turns.

    The loop also stops early when the run's token budget is used up. When either limit is hit,
    `limit_note` is appended and one last call is made with tools disabled, so the caller always
    gets text back. With `dedupe`, a call identical to an earlier one in this loop is not run again.
    Mutates `messages`. Raises LLMError on API failure.
    """
    turns = 0
    seen_calls: set[str] = set()
    while True:
        out_of_turns = turns >= max_turns or llm.over_budget()
        try:
            response = llm.call(phase, system, messages, tool_choice="none" if out_of_turns else "auto")
        except ToolCallRejected as exc:
            # The model asked for a tool that doesn't exist (or sent a malformed call). Count it as a
            # failed tool call and tell the model, the same way a tool error would be reported.
            name = exc.tool_name or "invalid_tool_call"
            llm.trace.tool_call(phase, llm.calls, name, {}, False, exc.detail, False, 0.0)
            llm.on_event("tool", {"phase": phase, "step": llm.calls, "name": name, "input": {},
                                  "ok": False, "cached": False, "output": exc.detail})
            if out_of_turns:
                return LoopResult("", "max_steps", turns)
            turns += 1
            note = (f"Your last tool call was rejected: {exc.detail}. "
                    f"The only tools are {', '.join(TOOL_NAMES)}. Use one of them, or answer.")
            if turns >= max_turns or llm.over_budget():
                note += "\n\n" + limit_note
            _append_note(messages, note)
            continue
        text = text_of(response)

        if response.stop_reason in ("refusal", "max_tokens"):
            return LoopResult(text, response.stop_reason, turns)
        if response.stop_reason != "tool_use" or out_of_turns:
            return LoopResult(text, "max_steps" if out_of_turns else "answered", turns)

        turns += 1
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_input = dict(block.input)
            key = f"{block.name}:{json.dumps(tool_input, sort_keys=True)}"
            started = time.perf_counter()
            if dedupe and key in seen_calls:
                result = ToolResult(ok=False, content=DUPLICATE_NOTE, blocked=True)
            else:
                result = toolbox.run(block.name, tool_input)
            seen_calls.add(key)
            llm.trace.tool_call(phase, llm.calls, block.name, tool_input, result.ok, result.content,
                                result.cached, time.perf_counter() - started, blocked=result.blocked)
            llm.on_event("tool", {"phase": phase, "step": llm.calls, "name": block.name, "input": tool_input,
                                  "ok": result.ok, "cached": result.cached, "output": result.content,
                                  "blocked": result.blocked})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result.content,
                "is_error": not result.ok,
            })

        if turns >= max_turns or llm.over_budget():
            tool_results.append({"type": "text", "text": limit_note})
        messages.append({"role": "user", "content": tool_results})
