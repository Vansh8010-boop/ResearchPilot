import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from research_pilot.agent import PlanExecuteAgent, ReActAgent
from research_pilot.config import load_settings
from research_pilot.providers import (
    OpenAICompatBackend,
    ProviderError,
    from_openai_response,
    to_openai_messages,
)
from research_pilot.tools import Toolbox


@pytest.fixture
def groq_settings(settings):
    return replace(settings, provider="groq", model="openai/gpt-oss-120b",
                   base_url="https://api.groq.test/openai/v1", api_key="gsk-test")


def chat(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10}}


def call(name, args, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedServer:
    """httpx transport that replies with scripted chat completions and records the requests."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

    def payload(self, i):
        return json.loads(self.requests[i].content)


def backend_for(settings, server, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return OpenAICompatBackend(settings, http=httpx.Client(transport=httpx.MockTransport(server)),
                               sleep=sleeps.append)


def test_message_translation():
    assistant_blocks = [SimpleNamespace(type="text", text="Let me check."),
                        SimpleNamespace(type="tool_use", id="call_1", name="calculator", input={"expression": "1+1"})]
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": assistant_blocks},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "bad", "is_error": True},
            {"type": "text", "text": "Step limit reached."},
        ]},
    ]
    out = to_openai_messages("sys", messages)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "q"}
    assert out[2]["content"] == "Let me check."
    assert out[2]["tool_calls"][0]["function"] == {"name": "calculator", "arguments": '{"expression": "1+1"}'}
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": "ERROR: bad"}
    assert out[4] == {"role": "user", "content": "Step limit reached."}


def test_response_translation():
    r = from_openai_response(chat(tool_calls=[call("web_search", {"query": "x"})], finish="tool_calls"))
    assert r.stop_reason == "tool_use"
    assert r.content[0].input == {"query": "x"}
    assert r.usage.input_tokens == 50

    bad = from_openai_response(chat(tool_calls=[{"id": "c", "function": {"name": "calculator", "arguments": "{oops"}}]))
    assert bad.content[0].input == {}
    assert from_openai_response(chat("partial", finish="length")).stop_reason == "max_tokens"
    assert from_openai_response(chat("done")).stop_reason == "end_turn"


def test_request_payload(groq_settings):
    server = ScriptedServer(chat("ok"), chat('{"steps": []}'))
    backend = backend_for(groq_settings, server)
    backend.create("sys", [{"role": "user", "content": "q"}], "auto", None)
    first = server.payload(0)
    assert server.requests[0].url == "https://api.groq.test/openai/v1/chat/completions"
    assert server.requests[0].headers["Authorization"] == "Bearer gsk-test"
    assert first["model"] == "openai/gpt-oss-120b" and first["tool_choice"] == "auto"
    assert {t["function"]["name"] for t in first["tools"]} == {"web_search", "fetch_page", "calculator"}

    backend.create("sys", [{"role": "user", "content": "q"}], None, {"type": "object"})
    second = server.payload(1)
    assert "tools" not in second
    assert second["response_format"] == {"type": "json_object"}
    assert "JSON object" in second["messages"][0]["content"]


def test_rate_limit_retry_and_daily_cap(groq_settings):
    sleeps = []
    server = ScriptedServer(httpx.Response(429, headers={"retry-after": "7"}, text="slow down"), chat("ok"))
    response = backend_for(groq_settings, server, sleeps).create("s", [{"role": "user", "content": "q"}], "auto", None)
    assert response.content[0].text == "ok" and sleeps == [7.0]

    server = ScriptedServer(httpx.Response(429, headers={"retry-after": "3600"}, text="daily limit"))
    with pytest.raises(ProviderError, match="retry after 3600s"):
        backend_for(groq_settings, server).create("s", [{"role": "user", "content": "q"}], "auto", None)

    server = ScriptedServer(httpx.Response(400, text="bad model"))
    with pytest.raises(ProviderError, match="HTTP 400"):
        backend_for(groq_settings, server).create("s", [{"role": "user", "content": "q"}], "auto", None)


def test_react_agent_over_openai_compat(groq_settings):
    server = ScriptedServer(
        chat(tool_calls=[call("calculator", {"expression": "6 * 7"})], finish="tool_calls"),
        chat("6 * 7 = 42."),
    )
    agent = ReActAgent(groq_settings, Toolbox(groq_settings), backend=backend_for(groq_settings, server))
    result = agent.run("What is 6 * 7?")
    assert result.stop_reason == "answered" and result.answer == "6 * 7 = 42."
    tool_message = server.payload(1)["messages"][-1]
    assert tool_message == {"role": "tool", "tool_call_id": "call_1", "content": "42"}


def test_plan_agent_over_openai_compat(groq_settings):
    plan = {"steps": [{"goal": "compute 6 * 7", "tool_hint": "calculator"}]}
    server = ScriptedServer(chat(json.dumps(plan)), chat("42 (calculator)"), chat("The answer is 42."))
    agent = PlanExecuteAgent(groq_settings, Toolbox(groq_settings), backend=backend_for(groq_settings, server))
    result = agent.run("What is 6 * 7?")
    assert result.plan == ["compute 6 * 7"]
    assert result.answer == "The answer is 42."


def test_provider_error_becomes_api_error(groq_settings):
    server = ScriptedServer(httpx.Response(429, headers={"retry-after": "3600"}, text="daily limit"))
    agent = ReActAgent(groq_settings, Toolbox(groq_settings), backend=backend_for(groq_settings, server))
    result = agent.run("q")
    assert result.stop_reason == "api_error" and result.answer is None


def test_provider_defaults(monkeypatch):
    for var in ("RP_PROVIDER", "RP_MODEL", "RP_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("research_pilot.config.load_dotenv", lambda *a, **k: None)
    groq = load_settings()
    assert groq.provider == "groq" and groq.model == "openai/gpt-oss-120b"
    ollama = load_settings(provider="ollama")
    assert ollama.base_url == "http://localhost:11434/v1" and ollama.api_key is None
    with pytest.raises(ValueError):
        load_settings(provider="nope")


def rejected_tool_call(tool="find"):
    return httpx.Response(400, json={"error": {
        "message": f"Tool call validation failed: attempted to call tool '{tool}' which was not in request.tools",
        "type": "invalid_request_error", "code": "tool_use_failed"}})


def test_backend_flags_rejected_tool_calls(groq_settings):
    from research_pilot.providers import ToolCallRejected
    server = ScriptedServer(rejected_tool_call())
    with pytest.raises(ToolCallRejected) as info:
        backend_for(groq_settings, server).create("s", [{"role": "user", "content": "q"}], "auto", None)
    assert info.value.tool_name == "find"


def test_agent_recovers_from_rejected_tool_call(groq_settings):
    server = ScriptedServer(
        rejected_tool_call(),
        chat(tool_calls=[call("calculator", {"expression": "6 * 7"})], finish="tool_calls"),
        chat("42"),
    )
    agent = ReActAgent(groq_settings, Toolbox(groq_settings), backend=backend_for(groq_settings, server))
    result = agent.run("What is 6 * 7?")
    assert result.stop_reason == "answered" and result.answer == "42"

    retry_messages = server.payload(1)["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "What is 6 * 7?" in retry_messages[-1]["content"]
    assert "rejected" in retry_messages[-1]["content"] and "web_search" in retry_messages[-1]["content"]

    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    tools = [e for e in trace["events"] if e["type"] == "tool_call"]
    assert [(t["name"], t["ok"]) for t in tools] == [("find", False), ("calculator", True)]
    assert trace["result"]["tool_calls"] == 2 and trace["result"]["tool_calls_ok"] == 1


def test_rejections_are_bounded(groq_settings):
    # Every call is rejected: the loop must still end (max_steps=3 turns, then one forced call).
    server = ScriptedServer(*[rejected_tool_call() for _ in range(groq_settings.max_steps + 1)])
    agent = ReActAgent(groq_settings, Toolbox(groq_settings), backend=backend_for(groq_settings, server))
    result = agent.run("q")
    assert result.stop_reason == "max_steps"
    assert len(server.requests) == groq_settings.max_steps + 1
