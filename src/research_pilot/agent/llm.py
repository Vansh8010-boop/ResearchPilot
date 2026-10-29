"""Shared pieces for every agent: the model call (with tracing) and a bounded tool-use loop."""

import time
from dataclasses import dataclass

from ..providers import ProviderError
from ..tools import Toolbox
from ..tracing import Trace
from .base import EventCallback


class LLMError(Exception):
    pass


def text_of(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


class LLM:
    """Wraps a provider backend: counts calls and records every call in the trace."""

    def __init__(self, backend, trace: Trace, on_event: EventCallback):
        self.backend = backend
        self.trace = trace
        self.on_event = on_event
        self.calls = 0

    def call(self, phase: str, system: str, messages: list, tool_choice: str | None = "auto",
             json_schema: dict | None = None):
        """tool_choice: "auto", "none" (tools defined but not callable) or None (no tools at all)."""
        self.calls += 1
        started = time.perf_counter()
        try:
            response = self.backend.create(system, messages, tool_choice, json_schema)
        except ProviderError as exc:
            self.trace.llm_call(phase, self.calls, time.perf_counter() - started, "api_error", None, str(exc))
            raise LLMError(str(exc)) from exc

        text = text_of(response)
        self.trace.llm_call(phase, self.calls, time.perf_counter() - started, response.stop_reason, response.usage, text)
        self.on_event("llm", {"phase": phase, "step": self.calls, "text": text, "stop_reason": response.stop_reason})
        return response


@dataclass
class LoopResult:
    text: str
    stop_reason: str  # answered | max_steps | refusal | max_tokens
    turns: int


def tool_loop(llm: LLM, toolbox: Toolbox, phase: str, system: str, messages: list,
              max_turns: int, limit_note: str) -> LoopResult:
    """Call the model, run the tools it asks for, feed results back; at most `max_turns` tool turns.

    When the limit is hit, `limit_note` is appended and one last call is made with tools disabled,
    so the caller always gets text back. Mutates `messages`. Raises LLMError on API failure.
    """
    turns = 0
    while True:
        out_of_turns = turns >= max_turns
        response = llm.call(phase, system, messages, tool_choice="none" if out_of_turns else "auto")
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
            started = time.perf_counter()
            result = toolbox.run(block.name, dict(block.input))
            llm.trace.tool_call(phase, llm.calls, block.name, dict(block.input), result.ok, result.content,
                                result.cached, time.perf_counter() - started)
            llm.on_event("tool", {"phase": phase, "step": llm.calls, "name": block.name, "input": dict(block.input),
                                  "ok": result.ok, "cached": result.cached, "output": result.content})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result.content,
                "is_error": not result.ok,
            })

        if turns >= max_turns:
            tool_results.append({"type": "text", "text": limit_note})
        messages.append({"role": "user", "content": tool_results})
