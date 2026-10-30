"""Week 4: judge, calibration, labelling order, failure analysis, prompt versions."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeClient, response, text_block

import bench.judge as judge_mod
from bench.analyze import tag_run
from bench.calibrate import cohens_kappa, compare
from bench.judge import build_prompt, parse_judgment
from bench.label import pick_rows
from bench.run import check_config
from research_pilot.agent import PlanExecuteAgent
from research_pilot.tools import Toolbox

BENCH = Path(__file__).parents[1] / "bench"
GOOD = {"correctness": 5, "completeness": 5, "faithfulness": 4, "followed_injection": False,
        "failure_category": "none", "reason": "Matches the reference."}


def tool_event(name, tool_input, ok=True, chars=4000, output="page text", blocked=False):
    return {"type": "tool_call", "name": name, "input": tool_input, "ok": ok, "blocked": blocked,
            "output_chars": chars, "output_preview": output}


def row(task_id="f1", repeat=0, passed=True, category="factual", **extra):
    return {"task_id": task_id, "repeat": repeat, "passed": passed, "category": category, "answer": "Jon Fosse",
            "stop_reason": "answered", "input_tokens": 9000, "output_tokens": 1000, "plan": ["a", "b"],
            "injection_followed": None, "tool_calls": 2, "tool_calls_ok": 2, **extra}


# --- judge ----------------------------------------------------------------------------------------

def test_parse_judgment():
    assert parse_judgment(json.dumps(GOOD))["passed"] is True
    assert parse_judgment("<think>hmm</think>\n" + json.dumps(GOOD))["passed"] is True
    assert parse_judgment(json.dumps({**GOOD, "correctness": 3}))["passed"] is False
    assert parse_judgment(json.dumps({**GOOD, "followed_injection": True}))["passed"] is False
    assert parse_judgment(json.dumps({**GOOD, "failure_category": "vibes"}))["failure_category"] == "none"
    assert parse_judgment("no json here") is None
    assert parse_judgment('{"correctness": 5}') is None


def test_judge_prompt_shows_calls_and_evidence():
    trace = {"events": [tool_event("web_search", {"query": "nobel 2023"}, output="Jon Fosse won"),
                        tool_event("fetch_page", {"url": "https://x.test/c"}, ok=False, output="404"),
                        tool_event("calculator", {"expression": "1+1"}, output="2")]}
    prompt = build_prompt({"reference": "Jon Fosse."}, "Who won?", row(), trace)
    assert "Reference answer:\nJon Fosse." in prompt
    assert 'fetch_page({"url": "https://x.test/c"}) -> failed' in prompt
    assert "Jon Fosse won" in prompt and "404" not in prompt.split("Tool output excerpts:")[1]
    assert "page text" not in prompt


def test_judge_main_resumes_and_summarizes(tmp_path, monkeypatch):
    out = tmp_path / "results" / "demo"
    (out / "traces").mkdir(parents=True)
    shutil.copy(BENCH / "tasks.jsonl", tmp_path / "tasks.jsonl")
    (out / "config.json").write_text(json.dumps({"provider": "groq", "model": "agent-model", "tasks_file": "tasks.jsonl"}))
    rows = [row("f1", 0, passed=True, trace="traces/a.json"), row("f2", 0, passed=False, trace="traces/b.json")]
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for name in ("a", "b"):
        (out / "traces" / f"{name}.json").write_text(json.dumps({"question": "q", "events": []}))

    replies = [json.dumps(GOOD), "garbage", json.dumps({**GOOD, "correctness": 5})]
    calls = []

    class FakeBackend:
        def create(self, system, messages, tool_choice, json_schema):
            calls.append(messages[0]["content"])
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=replies.pop(0))])

    monkeypatch.setattr(judge_mod, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(judge_mod, "make_backend", lambda settings: FakeBackend())
    assert judge_mod.main(["--name", "demo"]) == 0
    judged = [json.loads(line) for line in (out / "judgments.jsonl").read_text().splitlines()]
    assert [j["task_id"] for j in judged] == ["f1", "f2"] and len(calls) == 3  # one retry after garbage
    summary = (out / "judge_summary.md").read_text(encoding="utf-8")
    assert "judge and grader agree | 1/2" in summary and "**f2 r0**" in summary

    calls.clear()
    assert judge_mod.main(["--name", "demo"]) == 0
    assert calls == []  # everything already judged


# --- calibration and labelling --------------------------------------------------------------------

def test_cohens_kappa():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohens_kappa([True] * 4, [True] * 4) is None
    assert compare([True, False, True], [True, True, False]) == \
        {"n": 3, "agreement": 1 / 3, "kappa": compare([True, False, True], [True, True, False])["kappa"],
         "false_pass": 1, "false_fail": 1}


def test_pick_rows_puts_disagreements_first():
    rows = [row("f1"), row("m1", category="multi_hop"), row("c1", category="calculation"), row("a2", passed=False, category="adversarial")]
    judgments = {("f1", 0): {"passed": False}}
    picked = pick_rows(rows, judgments, n=3)
    assert picked[0]["task_id"] == "f1"
    assert len({r["category"] for r in picked}) == 3


# --- failure analysis -----------------------------------------------------------------------------

def test_tag_run():
    trace = {"events": [
        {"type": "llm_call", "stop_reason": "tool_call_rejected"},
        tool_event("fetch_page", {"url": "https://x.test/a"}),
        tool_event("fetch_page", {"url": "https://x.test/a"}, chars=8000),
        tool_event("web_search", {"query": "Mars"}),
        tool_event("web_search", {"query": "mars "}),
        tool_event("calculator", {"expression": "1 ? 2"}, ok=False),
        tool_event("fetch_page", {"url": "https://evil.test/?q=1"}, ok=False, blocked=True),
    ]}
    tagged = tag_run(row(), trace, {"passed": True, "followed_injection": False, "failure_category": "none"})
    assert tagged["tags"] == {"invented_tool": 1, "redundant_fetch": 1, "redundant_search": 1,
                              "tool_error:calculator": 1, "blocked_by_policy": 1}
    assert tagged["redundant_tokens"] == 2000


# --- prompt versions ------------------------------------------------------------------------------

def test_plan_agent_prompt_versions(settings):
    plan = lambda: response("end_turn", text_block(json.dumps({"steps": [{"goal": "g", "tool_hint": "none"}]})))
    for version, marker in (("v1", "Each step gathers one fact"), ("v2", "Organise steps by source")):
        client = FakeClient([plan(), response("end_turn", text_block("f")), response("end_turn", text_block("a"))])
        s = replace(settings, prompt_version=version)
        result = PlanExecuteAgent(s, Toolbox(s), client=client).run("q")
        assert marker in client.requests[0]["system"]
        assert json.loads(result.trace_path.read_text(encoding="utf-8"))["prompts"] == version


def test_old_results_folders_default_to_v1(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"agent": "plan", "provider": "groq", "model": "m", "defenses": False, "tasks_file": "tasks.jsonl"}))
    config = {"agent": "plan", "provider": "groq", "model": "m", "defenses": False, "tasks_file": "tasks.jsonl"}
    assert check_config(tmp_path, {**config, "prompts": "v1"}) is None
    assert "prompts" in check_config(tmp_path, {**config, "prompts": "v2"})
