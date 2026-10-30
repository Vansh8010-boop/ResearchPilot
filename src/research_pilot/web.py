"""Web API for serverless hosting (Vercel).

Serverless instances can't hold a job queue between requests, so instead of the background jobs in
api.py, one request runs one agent and streams its progress back as newline-delimited JSON:

    GET  /api/health      {"live": true, "provider": ..., "model": ...}
    POST /api/research    {"question": "...", "agent": "plan", "prompts": "v1"}
                          with "Authorization: Bearer <password>"; streams one JSON object per line

Safe by default: live questions stay off unless a password and the model and search keys are all set,
and defenses and the SSRF block are always on (a request can't turn them off).
"""

import hmac
import json
import os
import queue
import tempfile
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Literal

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .api import AgentFactory, Job, default_agent_factory
from .config import Settings, load_settings

HOSTED_MODEL = "openai/gpt-oss-20b"  # its own Groq allowance, so the benchmark's gpt-oss-120b stays free
RUN_DEADLINE_S = 280  # finish cleanly before the platform's 300s function limit cuts the stream
PING_S = 15


class WebRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    agent: Literal["plan", "react"] = "plan"
    prompts: Literal["v1", "v2"] | None = None


def hosted_settings() -> Settings:
    tmp = Path(tempfile.gettempdir()) / "research-pilot"  # the only writable place on serverless hosts
    settings = load_settings(defenses=True, block_private_urls=True, cache_dir=tmp / "cache", trace_dir=tmp / "traces")
    if settings.provider == "groq" and not os.getenv("RP_MODEL"):
        settings = replace(settings, model=HOSTED_MODEL)
    return settings


def stream_run(job: Job, settings: Settings, agent_factory: AgentFactory,
               deadline_s: float = RUN_DEADLINE_S, ping_s: float = PING_S) -> Iterator[str]:
    """Run the agent on a worker thread and yield its progress events as JSON lines."""
    events: queue.Queue[dict] = queue.Queue()

    def on_event(kind: str, data: dict) -> None:
        before = len(job.events)
        job.record(kind, data)
        for event in job.events[before:]:
            events.put(event)

    def worker() -> None:
        try:
            result = agent_factory(job.agent, settings, on_event).run(job.question)
            if result.stop_reason == "api_error":
                events.put({"type": "error", "message": "The model provider returned an error, often its "
                                                        "free-tier rate limit. Try again later."})
                return
            trace = json.loads(result.trace_path.read_text(encoding="utf-8"))["result"]
            events.put({"type": "final", "answer": result.answer, "stop_reason": result.stop_reason,
                        "stats": {k: trace.get(k) for k in ("steps", "input_tokens", "output_tokens", "tool_calls",
                                                            "tool_calls_ok", "tool_calls_blocked", "wall_time_s")}})
        except Exception:
            traceback.print_exc()  # details go to the server log, never to visitors
            events.put({"type": "error", "message": "The agent hit an internal error."})

    threading.Thread(target=worker, daemon=True).start()
    deadline = time.monotonic() + deadline_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield json.dumps({"type": "error", "message": "This question ran out of time. Try a simpler one."}) + "\n"
            return
        try:
            event = events.get(timeout=min(ping_s, remaining))
        except queue.Empty:
            yield json.dumps({"type": "ping"}) + "\n"  # keeps the connection open during rate-limit waits
            continue
        yield json.dumps(event, ensure_ascii=False) + "\n"
        if event["type"] in ("final", "error"):
            return


def create_web_app(settings: Settings | None = None, agent_factory: AgentFactory = default_agent_factory,
                   password: str | None = None) -> FastAPI:
    settings = settings or hosted_settings()
    password = password if password is not None else os.getenv("RP_API_TOKEN") or None
    live = bool(password and settings.api_key and settings.tavily_api_key)
    app = FastAPI(title="ResearchPilot", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "live": live, "provider": settings.provider, "model": settings.model}

    @app.post("/api/research")
    def research(request: WebRequest, authorization: str | None = Header(default=None)) -> StreamingResponse:
        if not live:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "live questions are not configured")
        if not hmac.compare_digest((authorization or "").encode(), f"Bearer {password}".encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong password")
        prompts = request.prompts or settings.prompt_version
        job = Job(id="web", question=request.question.strip(), agent=request.agent, defenses=True, prompts=prompts)
        return StreamingResponse(stream_run(job, replace(settings, prompt_version=prompts), agent_factory),
                                 media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    return app
