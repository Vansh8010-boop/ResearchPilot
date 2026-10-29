"""Per-run JSON traces. Every LLM call and tool call is recorded; week 4 failure analysis reads these."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PREVIEW_CHARS = 2000


class Trace:
    def __init__(self, question: str, model: str, trace_dir: Path, agent: str):
        self.run_id = uuid.uuid4().hex[:10]
        self.trace_dir = trace_dir
        self.started = time.perf_counter()
        self.data = {
            "run_id": self.run_id,
            "agent": agent,
            "question": question,
            "model": model,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "events": [],
        }

    def llm_call(self, phase: str, step: int, latency_s: float, stop_reason: str, usage, text: str) -> None:
        self.data["events"].append({
            "type": "llm_call",
            "phase": phase,
            "step": step,
            "latency_s": round(latency_s, 3),
            "stop_reason": stop_reason,
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "text": text,
        })

    def tool_call(self, phase: str, step: int, name: str, tool_input: dict, ok: bool, output: str,
                  cached: bool, latency_s: float, blocked: bool = False) -> None:
        self.data["events"].append({
            "type": "tool_call",
            "phase": phase,
            "step": step,
            "name": name,
            "input": tool_input,
            "ok": ok,
            "blocked": blocked,
            "cached": cached,
            "latency_s": round(latency_s, 3),
            "output_chars": len(output),
            "output_preview": output[:PREVIEW_CHARS],
        })

    def note(self, key: str, value) -> None:
        self.data[key] = value

    def finish(self, answer: str | None, stop_reason: str, steps: int) -> Path:
        llm_events = [e for e in self.data["events"] if e["type"] == "llm_call"]
        tool_events = [e for e in self.data["events"] if e["type"] == "tool_call"]
        executed = [e for e in tool_events if not e.get("blocked")]
        self.data["result"] = {
            "answer": answer,
            "stop_reason": stop_reason,
            "steps": steps,
            "wall_time_s": round(time.perf_counter() - self.started, 3),
            "input_tokens": sum(e["input_tokens"] or 0 for e in llm_events),
            "output_tokens": sum(e["output_tokens"] or 0 for e in llm_events),
            # Blocked calls (URL policy, duplicates) never ran, so they are counted apart from tool success.
            "tool_calls": len(executed),
            "tool_calls_ok": sum(e["ok"] for e in executed),
            "tool_calls_blocked": len(tool_events) - len(executed),
        }
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.trace_dir / f"{stamp}_{self.run_id}.json"
        path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
