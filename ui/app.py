"""Streamlit front end for ResearchPilot.

    uv run streamlit run ui/app.py

Two modes:
- Recorded runs: replays real benchmark traces from public/demo/ (RP_DEMO_DIR). Costs nothing and works
  even when the API is down or password-protected.
- Ask a question: submits to the HTTP API (RP_API_URL, default http://localhost:8000). If the API
  requires a token, visitors type it as a password unless RP_UI_API_TOKEN is set for a private
  deployment. The UI deliberately does not read RP_API_TOKEN: in a single-container deployment the
  API's token is in the same environment, and reading it would let every visitor skip the password.
"""

import json
import os
import re
import time
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.getenv("RP_API_URL", "http://localhost:8000").rstrip("/")
UI_TOKEN = os.getenv("RP_UI_API_TOKEN") or None
DEMO_DIR = Path(os.getenv("RP_DEMO_DIR", Path(__file__).resolve().parents[1] / "public" / "demo"))
POLL_SECONDS = 1.5
OUTPUT_PREVIEW = 300
URL_PATTERN = re.compile(r"https?://[^\s<>\"'`\)\]】]+")
TOOL_ICONS = {"web_search": "🔎", "fetch_page": "📄", "calculator": "🧮"}

st.set_page_config(page_title="ResearchPilot", page_icon="🔎", layout="wide")


# --- shared rendering -----------------------------------------------------------------------------

def sources_in(answer: str) -> list[str]:
    seen = []
    for url in URL_PATTERN.findall(answer or ""):
        url = url.rstrip(".,;:!?")
        if url not in seen:
            seen.append(url)
    return seen


def render_event(event: dict) -> None:
    if event["type"] == "tool":
        icon = TOOL_ICONS.get(event["name"], "🛠️")
        arg = next(iter(event["input"].values()), "") if event["input"] else ""
        mark = "⛔ blocked" if event["blocked"] else ("✓" if event["ok"] else "✗")
        cached = " · cached" if event["cached"] else ""
        st.markdown(f"{icon} `{event['name']}` {arg}  **{mark}**{cached}")
        if not event["ok"]:
            st.caption(event["output"])
    elif event["type"] == "finding":
        with st.expander(f"Step {event['step']} result: {event['goal']}"):
            st.markdown(event["finding"])
    elif event["type"] == "thought":
        st.caption(event["text"])


def render_job(job: dict, events: list[dict]) -> None:
    label = {"queued": "Waiting in queue", "running": "Researching", "done": "Done", "failed": "Failed"}[job["status"]]
    st.subheader(f"{label}: {job['question']}")
    st.caption(f"agent `{job['agent']}` · defenses {'on' if job['defenses'] else 'off'} · prompts `{job['prompts']}`")

    left, right = st.columns([3, 2])
    with right:
        if job["plan"]:
            st.markdown("**Plan**")
            st.markdown("\n".join(f"{i}. {goal}" for i, goal in enumerate(job["plan"], 1)))
        st.markdown("**Steps**")
        if not events:
            st.caption("No tool calls yet.")
        for event in events:
            if event["type"] != "plan":
                render_event(event)
    with left:
        if job["status"] in ("queued", "running"):
            st.info("Working on it. Runs usually take 30 seconds to 2 minutes on the free tier.")
        if job["error"]:
            st.error(job["error"])
        if job["answer"]:
            st.markdown("**Answer**")
            st.markdown(job["answer"])
            sources = sources_in(job["answer"])
            if sources:
                st.markdown("**Sources**")
                st.markdown("\n".join(f"- [{url}]({url})" for url in sources))
        if job["stats"]:
            s = job["stats"]
            cols = st.columns(5)
            cols[0].metric("Model calls", s.get("steps") or 0)
            cols[1].metric("Tool calls", f"{s.get('tool_calls_ok') or 0}/{s.get('tool_calls') or 0} ok")
            cols[2].metric("Blocked", s.get("tool_calls_blocked") or 0)
            cols[3].metric("Tokens", f"{(s.get('input_tokens') or 0) + (s.get('output_tokens') or 0):,}")
            cols[4].metric("Time", f"{s.get('wall_time_s') or 0:.0f}s")


# --- recorded runs --------------------------------------------------------------------------------

def trace_to_job(trace: dict) -> tuple[dict, list[dict]]:
    """Rebuild the live view (plan, tool calls, step findings, answer) from a saved trace."""
    events: list[dict] = []
    findings = trace.get("findings") or []
    current_step = None

    def close_step():
        if current_step and current_step.startswith("step"):
            index = int(current_step[4:])
            if index <= len(findings):
                goal = trace["plan"][index - 1] if index <= len(trace.get("plan", [])) else ""
                events.append({"type": "finding", "step": index, "goal": goal, "finding": findings[index - 1]})

    for e in trace.get("events", []):
        if e.get("phase") != current_step:
            close_step()
            current_step = e.get("phase")
        if e["type"] == "tool_call":
            events.append({"type": "tool", "phase": e["phase"], "name": e["name"], "input": e["input"],
                           "ok": e["ok"], "blocked": e.get("blocked", False), "cached": e.get("cached", False),
                           "output": e["output_preview"][:OUTPUT_PREVIEW]})
        elif e["stop_reason"] == "tool_use" and e.get("text"):
            events.append({"type": "thought", "phase": e["phase"], "text": e["text"][:500]})
    close_step()

    result = trace["result"]
    job = {
        "question": trace["question"], "agent": trace.get("agent", "plan"), "defenses": trace.get("defenses", False),
        "prompts": trace.get("prompts", "v1"), "status": "done", "plan": trace.get("plan", []),
        "answer": result["answer"], "error": None,
        "stats": {k: result.get(k) for k in ("steps", "input_tokens", "output_tokens", "tool_calls", "tool_calls_ok",
                                             "tool_calls_blocked", "wall_time_s")},
    }
    return job, events


def recorded_runs() -> None:
    index_path = DEMO_DIR / "index.json"
    if not index_path.exists():
        st.warning(f"No recorded runs found in {DEMO_DIR}.")
        return
    runs = json.loads(index_path.read_text(encoding="utf-8"))
    st.markdown("Real runs from the benchmark, replayed from their traces. Pick one to see what the agent did "
                "step by step.")
    choice = st.selectbox("Recorded run", range(len(runs)), format_func=lambda i: runs[i]["title"])
    run = runs[choice]
    st.info(run["description"])
    trace = json.loads((DEMO_DIR / run["file"]).read_text(encoding="utf-8"))
    job, events = trace_to_job(trace)
    render_job(job, events)


# --- live questions -------------------------------------------------------------------------------

def api(method: str, path: str, token: str | None = None, **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=15.0, **kwargs)
    response.raise_for_status()
    return response.json()


def live_questions() -> None:
    try:
        health = api("GET", "/health")
    except httpx.HTTPError as exc:
        st.error(f"The live agent isn't reachable right now ({exc}). Recorded runs still work.")
        return

    token = UI_TOKEN
    with st.sidebar:
        st.success(f"Live agent · {health['provider']} `{health['model']}`")
        if health.get("remaining_today") is not None:
            st.caption(f"{health['remaining_today']} of {health['daily_limit']} questions left today (UTC)")
        if health["auth"] and not token:
            token = st.text_input("Password", type="password", help="Live questions need a password.") or None
        agent = st.radio("Agent", ["plan", "react"], horizontal=True,
                         help="plan: planner, per-step executor, writer. react: one think-act loop.")
        defenses = st.toggle("Injection defenses", value=True,
                             help="Untrusted-content wrapping, URL provenance, link sanitizing, retries, loop limits.")
        prompts = st.selectbox("Prompt version", ["v1", "v2"],
                               index=["v1", "v2"].index(health.get("default_prompts", "v1")),
                               disabled=agent != "plan", help="Only the plan agent has prompt versions.")

    if health["auth"] and not token:
        st.info("Enter the password in the sidebar to ask live questions, or switch to Recorded runs.")
        return

    question = st.text_area("Research question", height=90,
                            placeholder="e.g. How many times larger is Canada than the UK by area?")
    if st.button("Research", type="primary", disabled=len(question.strip()) < 3):
        try:
            created = api("POST", "/research", token, json={"question": question, "agent": agent,
                                                            "defenses": defenses,
                                                            "prompts": prompts if agent == "plan" else None})
            st.session_state.job_id, st.session_state.events = created["id"], []
        except httpx.HTTPStatusError as exc:
            message = {401: "Wrong password.", 429: exc.response.json().get("detail", "Too many requests.")}
            st.error(message.get(exc.response.status_code, f"Couldn't start the job: {exc.response.text}"))

    if job_id := st.session_state.get("job_id"):
        events: list[dict] = st.session_state.setdefault("events", [])
        placeholder = st.empty()
        while True:
            try:
                job = api("GET", f"/research/{job_id}", token, params={"since": len(events)})
            except httpx.HTTPStatusError:
                st.session_state.pop("job_id")  # e.g. the server restarted and forgot the job
                break
            events.extend(job["events"])
            with placeholder.container():
                render_job(job, events)
            if job["status"] in ("done", "failed"):
                break
            time.sleep(POLL_SECONDS)


with st.sidebar:
    st.title("ResearchPilot")
    st.caption("A plan-and-execute research agent with prompt-injection defenses. "
               "[Source on GitHub](https://github.com/Vansh8010-boop/ResearchPilot)")
    mode = st.radio("Mode", ["Recorded runs", "Ask a question"],
                    help="Recorded runs replay real benchmark traces and cost nothing.")
    st.divider()

if mode == "Recorded runs":
    recorded_runs()
else:
    live_questions()
