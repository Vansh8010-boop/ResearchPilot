import json

from conftest import FakeClient, response, text_block, tool_block

from research_pilot.agent import PlanExecuteAgent
from research_pilot.tools import Toolbox


def plan_response(*goals):
    steps = [{"goal": g, "tool_hint": "web_search"} for g in goals]
    return response("end_turn", text_block(json.dumps({"steps": steps})))


def run_agent(settings, responses, question="How tall is X in feet?"):
    client = FakeClient(responses)
    agent = PlanExecuteAgent(settings, Toolbox(settings), client=client)
    return agent.run(question), client


def test_plan_execute_synthesize(settings):
    result, client = run_agent(settings, [
        plan_response("Find X's height in metres", "Convert it to feet"),
        response("end_turn", text_block("X is 100 m tall (https://example.com/x)")),
        response("tool_use", tool_block("calculator", {"expression": "100 * 3.28084"})),
        response("end_turn", text_block("328.084 ft")),
        response("end_turn", text_block("X is about 328 feet tall.")),
    ])
    assert result.stop_reason == "answered"
    assert result.answer == "X is about 328 feet tall."
    assert result.plan == ["Find X's height in metres", "Convert it to feet"]
    assert result.steps == 5

    planner, step1, step2a, step2b, synth = client.requests
    assert "tools" not in planner and planner["output_config"]["format"]["type"] == "json_schema"
    assert "tools" in step1
    # Step 2 sees step 1's finding; the synthesizer sees both and has no tools.
    assert "X is 100 m tall" in step2a["messages"][0]["content"]
    assert "<-- current step" in step2a["messages"][0]["content"]
    assert "328.084 ft" in synth["messages"][0]["content"] and "tools" not in synth

    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert [e["phase"] for e in trace["events"] if e["type"] == "llm_call"] == \
        ["plan", "step1", "step2", "step2", "synthesize"]
    assert trace["findings"] == ["X is 100 m tall (https://example.com/x)", "328.084 ft"]


def test_unusable_plan_falls_back_to_question(settings):
    result, client = run_agent(settings, [
        response("end_turn", text_block("not json")),
        response("end_turn", text_block("finding")),
        response("end_turn", text_block("answer")),
    ], question="What is Y?")
    assert result.plan == ["What is Y?"]
    assert result.answer == "answer"


def test_plan_is_capped(settings):
    goals = [f"step {i}" for i in range(settings.max_plan_steps + 3)]
    findings = [response("end_turn", text_block("f")) for _ in range(settings.max_plan_steps)]
    result, _ = run_agent(settings, [plan_response(*goals), *findings, response("end_turn", text_block("a"))])
    assert len(result.plan) == settings.max_plan_steps


def test_tool_budget_skips_remaining_steps(settings):
    from dataclasses import replace
    settings = replace(settings, max_tool_turns=1)
    result, client = run_agent(settings, [
        plan_response("first", "second"),
        response("tool_use", tool_block("calculator", {"expression": "1 + 1"})),
        response("end_turn", text_block("partial finding")),  # forced call after the limit note
        response("end_turn", text_block("best effort")),
    ])
    assert result.stop_reason == "max_steps"
    assert result.answer == "best effort"
    forced = client.requests[2]
    assert forced["tool_choice"] == {"type": "none"}
    assert "NOT FOUND: skipped" in client.requests[3]["messages"][0]["content"]


def test_planner_refusal(settings):
    result, _ = run_agent(settings, [response("refusal")])
    assert result.stop_reason == "refusal"
    assert result.answer is None
