"""Week 1 agent: a plain ReAct loop (think -> call tools -> observe) with a hard step limit.

Kept as a baseline to compare the plan-and-execute agent against.
"""

import anthropic

from ..config import Settings
from ..providers import make_backend
from ..tools import Toolbox
from ..tracing import Trace
from .base import AgentResult, EventCallback
from .llm import LLM, LLMError, tool_loop

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

STEP_LIMIT_NOTE = (
    "Step limit reached. Do not call any more tools. "
    "Answer now using only the information gathered so far, and say what is still unverified."
)


class ReActAgent:
    name = "react"

    def __init__(self, settings: Settings, toolbox: Toolbox, client: anthropic.Anthropic | None = None,
                 on_event: EventCallback | None = None, backend=None):
        self.settings = settings
        self.toolbox = toolbox
        self.backend = backend or make_backend(settings, client)
        self.on_event = on_event or (lambda kind, data: None)

    def run(self, question: str) -> AgentResult:
        trace = Trace(question, f"{self.settings.provider}:{self.settings.model}", self.settings.trace_dir, self.name)
        llm = LLM(self.backend, trace, self.on_event)
        messages: list = [{"role": "user", "content": question}]
        try:
            loop = tool_loop(llm, self.toolbox, "react", SYSTEM_PROMPT, messages,
                             self.settings.max_steps, STEP_LIMIT_NOTE)
            answer, stop_reason = loop.text or None, loop.stop_reason
        except LLMError:
            answer, stop_reason = None, "api_error"
        self.on_event("final", {"answer": answer, "stop_reason": stop_reason, "steps": llm.calls})
        return AgentResult(answer, stop_reason, llm.calls, trace.finish(answer, stop_reason, llm.calls))
