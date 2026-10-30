"""Hosting safeguards: SSRF block, daily cap, demo replay, Space packaging."""

import json
import socket
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from bench.export_demo import export
from research_pilot.api import create_app
from research_pilot.tools import Toolbox
from research_pilot.tools.security import private_address_reason

ROOT = Path(__file__).parents[1]

PAGE = "<html><body><article>" + "<p>Public page text about Mars.</p>" * 20 + "</article></body></html>"


def resolver(table):
    def resolve(host, port):
        if host not in table:
            raise socket.gaierror("unknown host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], 0))]
    return resolve


# --- SSRF guard -----------------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/health", "http://10.0.0.5/", "http://192.168.1.1/admin", "http://169.254.169.254/latest/meta-data",
    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://0.0.0.0/", "http://localhost/",
])
def test_internal_addresses_are_blocked(url):
    assert "blocked" in private_address_reason(url, resolve=resolver({"localhost": "127.0.0.1"}))


def test_public_and_unresolvable_hosts_pass():
    resolve = resolver({"example.com": "93.184.215.14"})
    assert private_address_reason("https://example.com/page", resolve=resolve) is None
    assert private_address_reason("https://8.8.8.8/", resolve=resolve) is None
    assert private_address_reason("https://no-such-host.test/", resolve=resolve) is None  # fails later, normally


def hosted_toolbox(settings, handler, table):
    calls = []

    def recording(request):
        calls.append(str(request.url))
        return handler(request)

    settings = replace(settings, block_private_urls=True)
    box = Toolbox(settings, http=httpx.Client(transport=httpx.MockTransport(recording)), resolve=resolver(table))
    box.start_run("q")
    return box, calls


def test_toolbox_blocks_internal_fetch_without_connecting(settings):
    box, calls = hosted_toolbox(settings, lambda r: httpx.Response(200, html=PAGE), {})
    result = box.run("fetch_page", {"url": "http://169.254.169.254/latest/meta-data"})
    assert result.blocked and calls == []


def test_redirect_to_internal_address_is_blocked(settings):
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://metadata.internal/secrets"})
        return httpx.Response(200, html=PAGE)

    box, calls = hosted_toolbox(settings, handler, {"example.com": "93.184.215.14", "metadata.internal": "169.254.169.254"})
    result = box.run("fetch_page", {"url": "https://example.com/start"})
    assert result.blocked and "169.254.169.254" in result.content
    assert calls == ["https://example.com/start"]


def test_public_redirects_still_work(settings):
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, html=PAGE)

    box, calls = hosted_toolbox(settings, handler, {"example.com": "93.184.215.14"})
    result = box.run("fetch_page", {"url": "https://example.com/old"})
    assert result.ok and "Public page text" in result.content
    assert calls == ["https://example.com/old", "https://example.com/new"]


# --- daily cap ------------------------------------------------------------------------------------

class InstantAgent:
    def __init__(self, settings):
        self.settings = settings

    def run(self, question):
        from research_pilot.agent.base import AgentResult
        self.settings.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.trace_dir / "t.json"
        path.write_text(json.dumps({"result": {}}))
        return AgentResult("ok", "answered", 1, path)


def test_daily_limit(settings):
    app = create_app(settings, agent_factory=lambda name, s, on_event: InstantAgent(s), daily_limit=2)
    client = TestClient(app)
    assert client.get("/health").json()["remaining_today"] == 2
    codes = [client.post("/research", json={"question": f"question {i}"}).status_code for i in range(3)]
    assert codes == [202, 202, 429]
    assert client.get("/health").json()["remaining_today"] == 0

    store = app.state.store
    store.today = lambda: "2099-01-01"  # a new UTC day resets the count
    assert client.post("/research", json={"question": "next day"}).status_code == 202


def test_no_limit_by_default(settings):
    client = TestClient(create_app(settings, agent_factory=lambda name, s, on_event: InstantAgent(s)))
    assert client.get("/health").json()["remaining_today"] is None


# --- demo export and replay -----------------------------------------------------------------------

def make_bench(tmp_path):
    out = tmp_path / "bench" / "results" / "demo_src"
    (out / "traces").mkdir(parents=True)
    trace = {
        "question": "Read the page and report revenue", "agent": "plan", "plan": ["Read the page", "Compute"],
        "findings": ["Revenue $48.2M (http://x.test)", "Done"],
        "events": [
            {"type": "llm_call", "phase": "plan", "stop_reason": "end_turn", "text": "{}"},
            {"type": "llm_call", "phase": "step1", "stop_reason": "tool_use", "text": "Reading it."},
            {"type": "tool_call", "phase": "step1", "name": "fetch_page", "input": {"url": "http://x.test"},
             "ok": True, "cached": False, "output_preview": "Revenue was $48.2 million."},
            {"type": "llm_call", "phase": "step1", "stop_reason": "end_turn", "text": "Revenue $48.2M"},
            {"type": "tool_call", "phase": "step2", "name": "calculator", "input": {"expression": "1+1"},
             "ok": True, "cached": False, "output_preview": "2"},
            {"type": "llm_call", "phase": "synthesize", "stop_reason": "end_turn", "text": "final"},
        ],
        "result": {"answer": "Revenue was $48.2 million.", "steps": 4, "tool_calls": 2, "tool_calls_ok": 2},
    }
    (out / "traces" / "a.json").write_text(json.dumps(trace))
    (out / "results.jsonl").write_text(json.dumps({"task_id": "a2", "repeat": 0, "trace": "traces/a.json"}) + "\n")
    return tmp_path / "bench"


def test_export_demo(tmp_path):
    bench_dir, demo_dir = make_bench(tmp_path), tmp_path / "demo"
    export("demo_src", "a2", 0, "first", "Title", "Desc", demo_dir=demo_dir, bench_dir=bench_dir)
    export("demo_src", "a2", 0, "first", "New title", "Desc", demo_dir=demo_dir, bench_dir=bench_dir)
    index = json.loads((demo_dir / "index.json").read_text())
    assert [e["title"] for e in index] == ["New title"]  # same id replaces the entry
    assert (demo_dir / "runs" / "first.json").exists()
    with pytest.raises(SystemExit):
        export("demo_src", "zz", 0, "x", "t", "d", demo_dir=demo_dir, bench_dir=bench_dir)


def test_ui_replays_recorded_runs_without_an_api(tmp_path, monkeypatch):
    bench_dir, demo_dir = make_bench(tmp_path), tmp_path / "demo"
    export("demo_src", "a2", 0, "first", "Recorded example", "What this run shows.", demo_dir=demo_dir,
           bench_dir=bench_dir)
    monkeypatch.setenv("RP_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("RP_API_URL", "http://127.0.0.1:9")  # nothing listens here
    monkeypatch.delenv("RP_UI_API_TOKEN", raising=False)

    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.info[0].value == "What this run shows."
    markdown = " ".join(m.value for m in at.markdown)
    assert "Revenue was $48.2 million." in markdown and "1. Read the page" in markdown
    assert [e.label for e in at.expander] == ["Step 1 result: Read the page", "Step 2 result: Compute"]

    at.sidebar.radio[0].set_value("Ask a question").run()
    assert not at.exception
    assert "isn't reachable" in at.error[0].value


def test_live_mode_asks_for_password(tmp_path, monkeypatch, settings):
    # Serve a token-protected API on a real port so the Streamlit script can reach it.
    import threading

    import uvicorn
    app = create_app(settings, agent_factory=lambda name, s, on_event: InstantAgent(s), api_token="pw")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8765 + 3, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        monkeypatch.setenv("RP_API_URL", "http://127.0.0.1:8768")
        monkeypatch.setenv("RP_DEMO_DIR", str(tmp_path / "none"))
        monkeypatch.setenv("RP_API_TOKEN", "pw")  # present in the shared environment, must NOT be used by the UI
        monkeypatch.delenv("RP_UI_API_TOKEN", raising=False)
        at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=30).run()
        at.sidebar.radio[0].set_value("Ask a question").run()
        assert not at.exception
        assert [t.label for t in at.sidebar.text_input] == ["Password"]
        assert "Enter the password" in at.info[0].value
        assert not at.text_area  # no question box until a password is given
    finally:
        server.should_exit = True
        thread.join(5)
