"""Week 2 agent: plan-and-execute.

1. Planner: one call, no tools, returns a JSON list of steps (structured output).
2. Executor: each step runs in its own short tool loop that sees the question, the plan and the
   findings so far, and reports back a finding with sources.
3. Synthesizer: one call, no tools, writes the final cited answer from the findings.

Budgets: at most `max_step_turns` tool turns per step and `max_tool_turns` across the whole run;
with defenses on, also `max_run_tokens` tokens per run.
"""

import json

import anthropic

from ..config import Settings
from ..providers import make_backend
from ..tools import Toolbox
from ..tracing import Trace
from .base import FINDINGS_SECURITY_RULES, SECURITY_RULES, AgentResult, EventCallback, sanitize_answer
from .llm import LLM, LLMError, text_of, tool_loop

PLANNER_PROMPT = """You are the planner of a research agent. Write a short plan for answering the user's question.

The executor will carry out each step with these tools: web_search (find sources), fetch_page (read a page),
calculator (exact arithmetic). A separate writer composes the final answer from the step results.

Rules:
- Use as few steps as possible: 1 for a single fact, more only when the question needs several facts or a computation.
- At most {max_steps} steps.
- Each step gathers one fact or does one computation, and names exactly what it should produce.
- When a step depends on an earlier result, say so (e.g. "using the director found in step 1").
- Do not add a step for writing the final answer.
- If the question gives a specific URL, the first step should read it."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "tool_hint": {"type": "string", "enum": ["web_search", "fetch_page", "calculator", "none"]},
                },
                "required": ["goal", "tool_hint"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["steps"],
    "additionalProperties": False,
}

EXECUTOR_PROMPT = """You are the executor of a research agent. Carry out ONE step of a research plan using the tools.

- Do only the current step; other steps are handled separately.
- Prefer fetch_page on an authoritative source over trusting a search snippet.
- Do all arithmetic with the calculator.
- If a tool returns an error, adjust (rephrase, try another source) instead of repeating the same call.

When done, reply with the finding only: the exact facts or numbers, each followed by its source URL.
If you could not complete the step, start your reply with "NOT FOUND:" and say what you tried."""

SYNTHESIZER_PROMPT = """You are the writer of a research agent. Answer the user's question using only the findings provided.

- Start with a direct answer in one or two sentences.
- Then list the key facts used, each with its source URL taken from the findings.
- If findings disagree, say so. If something could not be found, say what is missing instead of guessing.
- Do not add facts that are not in the findings."""

STEP_LIMIT_NOTE = (
    "Tool limit for this step reached. Do not call any more tools. "
    "Report what you found so far, or start with NOT FOUND: if you have nothing."
)


def _format_plan(plan: list[str], current: int | None = None) -> str:
    lines = []
    for i, goal in enumerate(plan, 1):
        marker = "  <-- current step" if i == current else ""
        lines.append(f"{i}. {goal}{marker}")
    return "\n".join(lines)


def _format_findings(findings: list[str]) -> str:
    if not findings:
        return "(none yet)"
    return "\n\n".join(f"Step {i} result:\n{f}" for i, f in enumerate(findings, 1))


class PlanExecuteAgent:
    name = "plan"

    def __init__(self, settings: Settings, toolbox: Toolbox, client: anthropic.Anthropic | None = None,
                 on_event: EventCallback | None = None, backend=None):
        self.settings = settings
        self.toolbox = toolbox
        self.backend = backend or make_backend(settings, client)
        self.on_event = on_event or (lambda kind, data: None)

    def _plan(self, llm: LLM, question: str) -> list[str]:
        system = PLANNER_PROMPT.format(max_steps=self.settings.max_plan_steps)
        response = llm.call("plan", system, [{"role": "user", "content": question}], tool_choice=None,
                            json_schema=PLAN_SCHEMA)
        if response.stop_reason == "refusal":
            raise _Refused()
        try:
            steps = [s["goal"].strip() for s in json.loads(text_of(response))["steps"] if s["goal"].strip()]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            steps = []
        # An unusable plan degrades to a single step that is the question itself.
        return steps[: self.settings.max_plan_steps] or [question]

    def _execute(self, llm: LLM, question: str, plan: list[str]) -> tuple[list[str], bool]:
        findings: list[str] = []
        budget = self.settings.max_tool_turns
        defenses = self.settings.defenses
        out_of_tokens = False
        for index, goal in enumerate(plan, 1):
            if budget <= 0:
                findings.append("NOT FOUND: skipped, the run's tool budget was used up.")
                continue
            if llm.over_budget():
                out_of_tokens = True
                findings.append("NOT FOUND: skipped, the run's token budget was used up.")
                continue
            prompt = (
                f"Question: {question}\n\nPlan:\n{_format_plan(plan, index)}\n\n"
                f"Results so far:\n{_format_findings(findings)}\n\nCurrent step ({index}): {goal}"
            )
            system = EXECUTOR_PROMPT + ("\n\n" + SECURITY_RULES if defenses else "")
            loop = tool_loop(llm, self.toolbox, f"step{index}", system,
                             [{"role": "user", "content": prompt}],
                             min(self.settings.max_step_turns, budget), STEP_LIMIT_NOTE, dedupe=defenses)
            if loop.stop_reason == "refusal":
                raise _Refused()
            budget -= loop.turns
            findings.append(loop.text or "NOT FOUND: the executor returned no text.")
            self.on_event("finding", {"step": index, "goal": goal, "finding": findings[-1]})
        return findings, budget <= 0 or out_of_tokens

    def _synthesize(self, llm: LLM, question: str, plan: list[str], findings: list[str]):
        prompt = (
            f"Question: {question}\n\nPlan that was followed:\n{_format_plan(plan)}\n\n"
            f"Findings:\n{_format_findings(findings)}"
        )
        system = SYNTHESIZER_PROMPT + ("\n\n" + FINDINGS_SECURITY_RULES if self.settings.defenses else "")
        return llm.call("synthesize", system, [{"role": "user", "content": prompt}], tool_choice=None)

    def run(self, question: str) -> AgentResult:
        defenses = self.settings.defenses
        trace = Trace(question, f"{self.settings.provider}:{self.settings.model}", self.settings.trace_dir, self.name)
        trace.note("defenses", defenses)
        self.toolbox.start_run(question)
        llm = LLM(self.backend, trace, self.on_event,
                  token_budget=self.settings.max_run_tokens if defenses else None)
        plan: list[str] = []
        answer = None
        try:
            plan = self._plan(llm, question)
            trace.note("plan", plan)
            self.on_event("plan", {"plan": plan})
            findings, out_of_budget = self._execute(llm, question, plan)
            trace.note("findings", findings)
            response = self._synthesize(llm, question, plan, findings)
            answer = text_of(response) or None
            if response.stop_reason in ("refusal", "max_tokens"):
                stop_reason = response.stop_reason
            else:
                stop_reason = "max_steps" if out_of_budget else "answered"
        except _Refused:
            stop_reason = "refusal"
        except LLMError:
            stop_reason = "api_error"
        if defenses:
            answer = sanitize_answer(answer, self.toolbox.provenance, trace)

        self.on_event("final", {"answer": answer, "stop_reason": stop_reason, "steps": llm.calls})
        return AgentResult(answer, stop_reason, llm.calls, trace.finish(answer, stop_reason, llm.calls), plan)


class _Refused(Exception):
    pass
