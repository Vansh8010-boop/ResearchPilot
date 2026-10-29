"""Week 1 agent: a plain ReAct loop (think -> call tools -> observe) with a hard step limit.

Week 2 replaces this with plan-and-execute; this loop stays as the baseline to compare against.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anthropic

from ..config import Settings
from ..tools import TOOL_SCHEMAS, Toolbox
from ..tracing import Trace

SYSTEM_PROMPT = """You are ResearchPilot, a research assistant that answers questions using tools.

Tools:
- web_search: find candidate sources.
- fetch_page: read a source in full. Prefer this over relying on search snippets.
- calculator: do all arithmetic here, never in your head.

How to work:
- Break the question into the facts you need, then gather each one with the tools.
- Prefer primary and authoritative sources. If sources disagree, say so.
- If a tool returns an error, adjust (rephrase the query, try another source) instead of repeating the same call.
- Stop calling tools once you have enough to answer.

Final answer format:
- Start with a direct answer in one or two sentences.
- Then list the key facts used, each with its source URL.
- If you could not find something, say what is missing instead of guessing."""

FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
STEP_LIMIT_NOTE = (
    "Step limit reached. Do not call any more tools. "
    "Answer now using only the information gathered so far, and say what is still unverified."
)

EventCallback = Callable[[str, dict], None]


@dataclass
class AgentResult:
    answer: str | None
    stop_reason: str  # answered | max_steps | refusal | max_tokens | api_error
    steps: int
    trace_path: Path


def _text_of(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


class ReActAgent:
    def __init__(self, settings: Settings, toolbox: Toolbox, client: anthropic.Anthropic | None = None,
                 on_event: EventCallback | None = None):
        self.settings = settings
        self.toolbox = toolbox
        self.client = client or anthropic.Anthropic()
        self.on_event = on_event or (lambda kind, data: None)

    def _call_llm(self, messages: list, allow_tools: bool = True):
        kwargs = {}
        if self.settings.model in FALLBACK_MODELS:
            # On a safety refusal the API retries on a fallback model instead of returning nothing.
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        return self.client.beta.messages.create(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            tool_choice={"type": "auto" if allow_tools else "none"},
            output_config={"effort": self.settings.effort},
            messages=messages,
            **kwargs,
        )

    def run(self, question: str) -> AgentResult:
        trace = Trace(question, self.settings.model, self.settings.trace_dir)
        messages: list = [{"role": "user", "content": question}]
        step = 0

        def finish(answer: str | None, stop_reason: str) -> AgentResult:
            self.on_event("final", {"answer": answer, "stop_reason": stop_reason, "steps": step})
            return AgentResult(answer, stop_reason, step, trace.finish(answer, stop_reason, step))

        while True:
            step += 1
            out_of_steps = step > self.settings.max_steps
            started = time.perf_counter()
            try:
                response = self._call_llm(messages, allow_tools=not out_of_steps)
            except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
                trace.llm_call(step, time.perf_counter() - started, "api_error", None, f"{type(exc).__name__}: {exc}")
                return finish(None, "api_error")

            text = _text_of(response)
            trace.llm_call(step, time.perf_counter() - started, response.stop_reason, response.usage, text)
            self.on_event("llm", {"step": step, "text": text, "stop_reason": response.stop_reason})

            if response.stop_reason in ("refusal", "max_tokens"):
                return finish(text or None, response.stop_reason)
            if response.stop_reason != "tool_use" or out_of_steps:
                return finish(text, "max_steps" if out_of_steps else "answered")

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                started = time.perf_counter()
                result = self.toolbox.run(block.name, dict(block.input))
                trace.tool_call(step, block.name, dict(block.input), result.ok, result.content,
                                result.cached, time.perf_counter() - started)
                self.on_event("tool", {"step": step, "name": block.name, "input": dict(block.input),
                                       "ok": result.ok, "cached": result.cached, "output": result.content})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.content,
                    "is_error": not result.ok,
                })

            if step == self.settings.max_steps:
                tool_results.append({"type": "text", "text": STEP_LIMIT_NOTE})
            messages.append({"role": "user", "content": tool_results})
