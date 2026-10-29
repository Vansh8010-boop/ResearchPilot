import json

from conftest import FakeClient, response, text_block, tool_block

from research_pilot.agent import ReActAgent
from research_pilot.tools import Toolbox


def run_agent(settings, responses, question="What is 6 * 7?"):
    client = FakeClient(responses)
    agent = ReActAgent(settings, Toolbox(settings), client=client)
    return agent.run(question), client


def test_answers_after_tool_call(settings):
    result, client = run_agent(settings, [
        response("tool_use", text_block("Let me compute."), tool_block("calculator", {"expression": "6 * 7"})),
        response("end_turn", text_block("6 * 7 = 42.")),
    ])
    assert result.stop_reason == "answered"
    assert result.answer == "6 * 7 = 42."
    assert result.steps == 2

    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result == {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42", "is_error": False}


def test_tool_errors_are_flagged_to_model(settings):
    _, client = run_agent(settings, [
        response("tool_use", tool_block("calculator", {"expression": "1 / 0"})),
        response("end_turn", text_block("Undefined.")),
    ])
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True


def test_step_limit_forces_final_answer(settings):
    looping = [response("tool_use", tool_block("calculator", {"expression": "1 + 1"})) for _ in range(settings.max_steps)]
    result, client = run_agent(settings, looping + [response("end_turn", text_block("Best effort answer."))])

    assert result.stop_reason == "max_steps"
    assert result.answer == "Best effort answer."
    assert len(client.requests) == settings.max_steps + 1
    final_request = client.requests[-1]
    assert final_request["tool_choice"] == {"type": "none"}
    assert final_request["messages"][-1]["content"][-1]["type"] == "text"


def test_refusal_stops_run(settings):
    result, _ = run_agent(settings, [response("refusal")])
    assert result.stop_reason == "refusal"
    assert result.answer is None


def test_trace_written(settings):
    result, _ = run_agent(settings, [
        response("tool_use", tool_block("calculator", {"expression": "6 * 7"})),
        response("end_turn", text_block("42")),
    ])
    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert [e["type"] for e in trace["events"]] == ["llm_call", "tool_call", "llm_call"]
    assert trace["result"]["tool_calls"] == 1
    assert trace["result"]["tool_calls_ok"] == 1
    assert trace["result"]["input_tokens"] == 200


def test_request_shape(settings):
    _, client = run_agent(settings, [response("end_turn", text_block("hi"))])
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    assert request["output_config"] == {"effort": settings.effort}
    assert {t["name"] for t in request["tools"]} == {"web_search", "fetch_page", "calculator"}
