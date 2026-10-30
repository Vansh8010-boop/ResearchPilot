"""Serverless web API: fail-closed setup, password, forced defenses, streamed progress."""

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from research_pilot.agent.base import AgentResult
from research_pilot.api import Job
from research_pilot.web import HOSTED_MODEL, create_web_app, hosted_settings, stream_run

ROOT = Path(__file__).parents[1]
AUTH = {"Authorization": "Bearer pw"}


class FakeAgent:
    def __init__(self, settings, on_event, stop_reason="answered", crash=False):
        self.settings, self.on_event, self.stop_reason, self.crash = settings, on_event, stop_reason, crash

    def run(self, question):
        if self.crash:
            raise RuntimeError("secret detail")
        self.on_event("plan", {"plan": ["find it"]})
        self.on_event("tool", {"phase": "step1", "step": 2, "name": "web_search", "input": {"query": question},
                               "ok": True, "cached": False, "output": "x", "blocked": False})
        self.settings.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.trace_dir / "t.json"
        path.write_text(json.dumps({"result": {"steps": 2, "tool_calls": 1, "tool_calls_ok": 1}}))
        return AgentResult("The answer.", self.stop_reason, 2, path)


@pytest.fixture
def live_settings(settings):
    return replace(settings, api_key="gsk-test", defenses=True, block_private_urls=True)


def client_for(settings, password="pw", **agent_kwargs):
    seen = {}

    def factory(name, job_settings, on_event):
        seen["agent"], seen["settings"] = name, job_settings
        return FakeAgent(job_settings, on_event, **agent_kwargs)

    return TestClient(create_web_app(settings, factory, password=password)), seen


def lines(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_health_reports_live(live_settings):
    client, _ = client_for(live_settings)
    assert client.get("/api/health").json()["live"] is True


@pytest.mark.parametrize("change", [{"api_key": None}, {"tavily_api_key": None}])
def test_missing_keys_keep_live_off(live_settings, change):
    client, _ = client_for(replace(live_settings, **change))
    assert client.get("/api/health").json()["live"] is False
    assert client.post("/api/research", json={"question": "what?"}, headers=AUTH).status_code == 503


def test_no_password_keeps_live_off(live_settings, monkeypatch):
    monkeypatch.delenv("RP_API_TOKEN", raising=False)
    client, seen = client_for(live_settings, password=None)
    assert client.get("/api/health").json()["live"] is False
    assert client.post("/api/research", json={"question": "what?"}).status_code == 503
    assert seen == {}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "pw"}])
def test_wrong_password(live_settings, headers):
    client, seen = client_for(live_settings)
    assert client.post("/api/research", json={"question": "what?"}, headers=headers).status_code == 401
    assert seen == {}


def test_stream_contains_progress_and_result(live_settings):
    client, seen = client_for(live_settings)
    response = client.post("/api/research", json={"question": "What is 6 * 7?", "agent": "react", "prompts": "v2"},
                           headers=AUTH)
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/x-ndjson")
    events = lines(response)
    assert [e["type"] for e in events] == ["plan", "tool", "final"]
    assert events[-1]["answer"] == "The answer." and events[-1]["stats"]["tool_calls"] == 1
    assert seen["agent"] == "react" and seen["settings"].prompt_version == "v2"


def test_request_cannot_turn_defenses_off(live_settings):
    client, seen = client_for(live_settings)
    client.post("/api/research", headers=AUTH, json={"question": "what?", "defenses": False,
                                                     "block_private_urls": False})
    assert seen["settings"].defenses is True and seen["settings"].block_private_urls is True


def test_hosted_settings_are_safe(monkeypatch):
    for name in ("RP_MODEL", "RP_PROVIDER", "RP_DEFENSES", "RP_BLOCK_PRIVATE_URLS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RP_DEFENSES", "off")
    monkeypatch.setenv("RP_BLOCK_PRIVATE_URLS", "off")
    monkeypatch.setattr("research_pilot.config.load_dotenv", lambda *a, **k: None)
    settings = hosted_settings()
    assert settings.defenses is True and settings.block_private_urls is True
    assert settings.model == HOSTED_MODEL
    assert "research-pilot" in str(settings.trace_dir)


def test_crash_details_stay_out_of_the_response(live_settings, capsys):
    client, _ = client_for(live_settings, crash=True)
    events = lines(client.post("/api/research", json={"question": "what?"}, headers=AUTH))
    assert events == [{"type": "error", "message": "The agent hit an internal error."}]
    assert "secret detail" in capsys.readouterr().err


def test_provider_error_is_reported(live_settings):
    client, _ = client_for(live_settings, stop_reason="api_error")
    events = lines(client.post("/api/research", json={"question": "what?"}, headers=AUTH))
    assert events[-1]["type"] == "error" and "rate limit" in events[-1]["message"]


def test_slow_runs_get_pings_then_a_deadline(live_settings):
    release = threading.Event()

    class Slow:
        def run(self, question):
            release.wait(5)
            raise RuntimeError("late")

    job = Job(id="t", question="q", agent="plan", defenses=True, prompts="v1")
    started = time.monotonic()
    events = [json.loads(line) for line in stream_run(job, live_settings, lambda *a: Slow(), deadline_s=0.3, ping_s=0.1)]
    release.set()
    assert time.monotonic() - started < 2
    assert {"type": "ping"} in events
    assert events[-1]["type"] == "error" and "ran out of time" in events[-1]["message"]


def test_page_is_safe_and_self_contained():
    html = (ROOT / "public" / "index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in html and "localStorage" not in html and "sessionStorage" not in html
    assert "<script src" not in html
    index = json.loads((ROOT / "public" / "demo" / "index.json").read_text(encoding="utf-8"))
    assert index and all((ROOT / "public" / "demo" / run["file"]).exists() for run in index)


def test_vercel_config():
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert config["functions"]["app.py"]["maxDuration"] == 300
