"""HTTP API: submit a research question as a background job, then poll it for progress.

    uv run uvicorn research_pilot.api:app --port 8000

    POST /research            {"question": "...", "agent": "plan", "defenses": true}  -> 202 {"id": ...}
    GET  /research/{id}?since=N   status, plan, answer, and the progress events after the first N
    GET  /research            recent jobs
    GET  /health              provider, model, defaults

Jobs run one at a time on a single worker thread: free-tier rate limits and a laptop's memory both
favour sequential runs. Set RP_API_TOKEN to require "Authorization: Bearer <token>".
"""

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from . import __version__
from .agent import AGENTS
from .config import Settings, load_settings
from .tools import Toolbox

MAX_QUEUED = 20
MAX_KEPT_JOBS = 200
OUTPUT_PREVIEW = 300
THOUGHT_PREVIEW = 500

AgentFactory = Callable[[str, Settings, Callable[[str, dict], None]], object]


def default_agent_factory(agent_name: str, settings: Settings, on_event):
    return AGENTS[agent_name](settings, Toolbox(settings), on_event=on_event)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    agent: Literal["plan", "react"] = "plan"
    defenses: bool = True
    prompts: Literal["v1", "v2"] | None = None


@dataclass
class Job:
    id: str
    question: str
    agent: str
    defenses: bool
    prompts: str
    status: str = "queued"  # queued | running | done | failed
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    plan: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    answer: str | None = None
    stop_reason: str | None = None
    stats: dict = field(default_factory=dict)
    error: str | None = None

    def record(self, kind: str, data: dict) -> None:
        """Agent progress callback: keep a compact, JSON-friendly copy of each event."""
        if kind == "plan":
            self.plan = list(data["plan"])
            self.events.append({"type": "plan", "plan": self.plan})
        elif kind == "tool":
            self.events.append({
                "type": "tool", "phase": data["phase"], "name": data["name"], "input": data["input"],
                "ok": data["ok"], "blocked": data.get("blocked", False), "cached": data["cached"],
                "output": data["output"][:OUTPUT_PREVIEW],
            })
        elif kind == "finding":
            self.events.append({"type": "finding", "step": data["step"], "goal": data["goal"],
                                "finding": data["finding"]})
        elif kind == "llm" and data["stop_reason"] == "tool_use" and data["text"]:
            self.events.append({"type": "thought", "phase": data["phase"], "text": data["text"][:THOUGHT_PREVIEW]})

    def view(self, since: int = 0) -> dict:
        return {
            "id": self.id, "question": self.question, "agent": self.agent, "defenses": self.defenses,
            "prompts": self.prompts, "status": self.status, "created": self.created, "started": self.started,
            "finished": self.finished, "plan": self.plan, "answer": self.answer, "stop_reason": self.stop_reason,
            "stats": self.stats, "error": self.error, "events_total": len(self.events),
            "events": self.events[since:],
        }


class JobStore:
    def __init__(self, settings: Settings, agent_factory: AgentFactory):
        self.settings = settings
        self.agent_factory = agent_factory
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research")

    def submit(self, request: ResearchRequest) -> Job:
        with self.lock:
            queued = sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))
            if queued >= MAX_QUEUED:
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, f"{queued} jobs already waiting; try later")
            job = Job(id=uuid.uuid4().hex[:12], question=request.question.strip(), agent=request.agent,
                      defenses=request.defenses, prompts=request.prompts or self.settings.prompt_version)
            self.jobs[job.id] = job
            while len(self.jobs) > MAX_KEPT_JOBS:
                oldest = next(iter(self.jobs))
                if self.jobs[oldest].status in ("queued", "running"):
                    break
                self.jobs.pop(oldest)
        self.executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job '{job_id}'")
        return job

    def _run(self, job: Job) -> None:
        job.status, job.started = "running", time.time()
        settings = replace(self.settings, defenses=job.defenses, prompt_version=job.prompts)
        try:
            agent = self.agent_factory(job.agent, settings, job.record)
            result = agent.run(job.question)
            job.answer, job.stop_reason = result.answer, result.stop_reason
            trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
            job.stats = {k: trace["result"].get(k) for k in ("steps", "input_tokens", "output_tokens", "tool_calls",
                                                             "tool_calls_ok", "tool_calls_blocked", "wall_time_s")}
            job.stats["trace"] = result.trace_path.name
            if result.stop_reason == "api_error":
                job.status, job.error = "failed", "the model provider returned an error (often a rate limit)"
            else:
                job.status = "done"
        except Exception as exc:  # report any crash on the job instead of losing it in the worker thread
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()


def create_app(settings: Settings | None = None, agent_factory: AgentFactory = default_agent_factory,
               api_token: str | None = None) -> FastAPI:
    settings = settings or load_settings()
    api_token = api_token if api_token is not None else os.getenv("RP_API_TOKEN") or None
    store = JobStore(settings, agent_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        store.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="ResearchPilot", version=__version__, lifespan=lifespan)
    app.state.store = store

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if api_token and authorization != f"Bearer {api_token}":
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or wrong bearer token")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "provider": settings.provider, "model": settings.model,
                "default_prompts": settings.prompt_version, "auth": bool(api_token)}

    @app.post("/research", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_token)])
    def submit(request: ResearchRequest) -> dict:
        job = store.submit(request)
        return {"id": job.id, "status": job.status}

    @app.get("/research", dependencies=[Depends(require_token)])
    def list_jobs(limit: int = Query(20, ge=1, le=MAX_KEPT_JOBS)) -> list[dict]:
        jobs = list(store.jobs.values())[-limit:][::-1]
        return [{"id": j.id, "question": j.question, "status": j.status, "created": j.created} for j in jobs]

    @app.get("/research/{job_id}", dependencies=[Depends(require_token)])
    def get_job(job_id: str, since: int = Query(0, ge=0)) -> dict:
        return store.get(job_id).view(since)

    return app


def __getattr__(name: str):
    # `uvicorn research_pilot.api:app` builds the app on first access, so importing this module
    # (e.g. in tests) doesn't read settings or start a worker thread.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
