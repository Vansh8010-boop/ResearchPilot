import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from research_pilot.agent.base import AgentResult
from research_pilot.api import MAX_QUEUED, OUTPUT_PREVIEW, create_app


class FakeAgent:
    """Emits the same callback events as the real agents, then writes a minimal trace."""

    def __init__(self, settings, on_event, gate=None, fail=False, stop_reason="answered"):
        self.settings, self.on_event, self.gate, self.fail, self.stop_reason = settings, on_event, gate, fail, stop_reason

    def run(self, question):
        if self.gate:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("boom")
        self.on_event("plan", {"plan": ["find it"]})
        self.on_event("llm", {"phase": "step1", "step": 2, "text": "Searching.", "stop_reason": "tool_use"})
        self.on_event("tool", {"phase": "step1", "step": 2, "name": "web_search", "input": {"query": question},
                               "ok": True, "cached": False, "output": "x" * 1000, "blocked": False})
        self.on_event("finding", {"step": 1, "goal": "find it", "finding": "42 (https://example.com)"})
        self.settings.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.trace_dir / "t.json"
        path.write_text(json.dumps({"result": {"steps": 3, "input_tokens": 100, "output_tokens": 20, "tool_calls": 1,
                                               "tool_calls_ok": 1, "tool_calls_blocked": 0, "wall_time_s": 1.5},
                                    "settings_seen": {"defenses": self.settings.defenses,
                                                      "prompts": self.settings.prompt_version}}))
        answer = None if self.stop_reason == "api_error" else "The answer is 42. Source: https://example.com"
        return AgentResult(answer, self.stop_reason, 3, path, ["find it"])


def make_client(settings, token=None, **agent_kwargs):
    seen = {}

    def factory(name, job_settings, on_event):
        seen["agent"], seen["settings"] = name, job_settings
        return FakeAgent(job_settings, on_event, **agent_kwargs)

    return TestClient(create_app(settings, agent_factory=factory, api_token=token)), seen


def wait_for(client, job_id, headers=None):
    for _ in range(200):
        job = client.get(f"/research/{job_id}", headers=headers or {}).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health(settings):
    client, _ = make_client(settings)
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["provider"] == "anthropic" and body["auth"] is False


def test_job_lifecycle(settings):
    client, seen = make_client(settings)
    created = client.post("/research", json={"question": "What is 6 * 7?", "defenses": False, "prompts": "v2"})
    assert created.status_code == 202
    job = wait_for(client, created.json()["id"])

    assert job["status"] == "done" and job["answer"].startswith("The answer is 42")
    assert job["plan"] == ["find it"]
    assert [e["type"] for e in job["events"]] == ["plan", "thought", "tool", "finding"]
    assert len(job["events"][2]["output"]) == OUTPUT_PREVIEW
    assert job["stats"]["tool_calls"] == 1 and job["stats"]["trace"] == "t.json"
    assert seen["agent"] == "plan"
    assert seen["settings"].defenses is False and seen["settings"].prompt_version == "v2"

    later = client.get(f"/research/{job['id']}", params={"since": 3}).json()
    assert [e["type"] for e in later["events"]] == ["finding"] and later["events_total"] == 4

    listed = client.get("/research").json()
    assert listed[0]["id"] == job["id"]


def test_defaults_are_safe(settings):
    client, seen = make_client(settings)
    wait_for(client, client.post("/research", json={"question": "anything"}).json()["id"])
    assert seen["settings"].defenses is True
    assert seen["settings"].prompt_version == settings.prompt_version


@pytest.mark.parametrize("body", [{"question": "hi"}, {"question": "x" * 2001}, {"question": "ok?", "agent": "nope"},
                                  {}])
def test_validation(settings, body):
    client, _ = make_client(settings)
    assert client.post("/research", json=body).status_code == 422


def test_unknown_job(settings):
    client, _ = make_client(settings)
    assert client.get("/research/nope").status_code == 404


def test_failures_are_reported(settings):
    client, _ = make_client(settings, fail=True)
    job = wait_for(client, client.post("/research", json={"question": "crash please"}).json()["id"])
    assert job["status"] == "failed" and "RuntimeError: boom" in job["error"]

    client, _ = make_client(settings, stop_reason="api_error")
    job = wait_for(client, client.post("/research", json={"question": "rate limited"}).json()["id"])
    assert job["status"] == "failed" and "rate limit" in job["error"]


def test_token_auth(settings):
    client, _ = make_client(settings, token="s3cret")
    assert client.post("/research", json={"question": "what?"}).status_code == 401
    assert client.get("/research", headers={"Authorization": "Bearer wrong"}).status_code == 401
    headers = {"Authorization": "Bearer s3cret"}
    created = client.post("/research", json={"question": "what?"}, headers=headers)
    assert created.status_code == 202
    assert wait_for(client, created.json()["id"], headers)["status"] == "done"
    assert client.get("/health").json()["auth"] is True  # health stays public


def test_queue_limit(settings):
    gate = threading.Event()  # hold the worker so jobs pile up
    client, _ = make_client(settings, gate=gate)
    codes = [client.post("/research", json={"question": f"question {i}"}).status_code for i in range(MAX_QUEUED + 1)]
    gate.set()
    assert codes[:MAX_QUEUED] == [202] * MAX_QUEUED and codes[-1] == 429
