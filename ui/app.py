"""Streamlit front end for the ResearchPilot API.

    uv run streamlit run ui/app.py

Talks only to the HTTP API (RP_API_URL, default http://localhost:8000), so it needs no model keys.
"""

import os
import re
import time

import httpx
import streamlit as st

API_URL = os.getenv("RP_API_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.getenv("RP_API_TOKEN") or None
POLL_SECONDS = 1.5
URL_PATTERN = re.compile(r"https?://[^\s<>\"'`\)\]】]+")
TOOL_ICONS = {"web_search": "🔎", "fetch_page": "📄", "calculator": "🧮"}

st.set_page_config(page_title="ResearchPilot", page_icon="🔎", layout="wide")


def api(method: str, path: str, **kwargs):
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    response = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=15.0, **kwargs)
    response.raise_for_status()
    return response.json()


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


with st.sidebar:
    st.title("ResearchPilot")
    try:
        health = api("GET", "/health")
        st.success(f"API connected · {health['provider']} `{health['model']}`")
    except httpx.HTTPError as exc:
        st.error(f"Can't reach the API at {API_URL}: {exc}")
        st.stop()
    agent = st.radio("Agent", ["plan", "react"], horizontal=True,
                     help="plan: planner, per-step executor, writer. react: one think-act loop.")
    defenses = st.toggle("Injection defenses", value=True,
                         help="Untrusted-content wrapping, URL provenance, link sanitizing, retries, loop limits.")
    prompts = st.selectbox("Prompt version", ["v1", "v2"],
                           index=["v1", "v2"].index(health.get("default_prompts", "v1")),
                           disabled=agent != "plan", help="Only the plan agent has prompt versions.")
    st.divider()
    st.markdown("**Recent questions**")
    for item in api("GET", "/research", params={"limit": 8}):
        if st.button(f"{item['question'][:48]} ({item['status']})", key=item["id"], use_container_width=True):
            st.session_state.job_id = item["id"]
            st.session_state.events = []

question = st.text_area("Research question", placeholder="e.g. How many times larger is Canada than the UK by area?",
                        height=90)
if st.button("Research", type="primary", disabled=len(question.strip()) < 3):
    try:
        created = api("POST", "/research", json={"question": question, "agent": agent, "defenses": defenses,
                                                 "prompts": prompts if agent == "plan" else None})
        st.session_state.job_id = created["id"]
        st.session_state.events = []
    except httpx.HTTPStatusError as exc:
        st.error(f"Couldn't start the job: {exc.response.text}")

if job_id := st.session_state.get("job_id"):
    events: list[dict] = st.session_state.setdefault("events", [])
    placeholder = st.empty()
    while True:
        job = api("GET", f"/research/{job_id}", params={"since": len(events)})
        events.extend(job["events"])
        with placeholder.container():
            render_job(job, events)
        if job["status"] in ("done", "failed"):
            break
        time.sleep(POLL_SECONDS)
