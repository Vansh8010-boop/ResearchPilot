"""Run the benchmark: every task x N repeats, sequentially, resumable.

    uv run python -m bench.run --agent plan --repeats 3
    uv run python -m bench.run --agent react --name react_baseline
    uv run python -m bench.run --tasks f1,c3 --repeats 1      # quick smoke test

Writes bench/results/<name>/: results.jsonl (one line per run), traces/, summary.json, summary.md.
Re-running with the same --name skips runs that already finished.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from research_pilot.agent import AGENTS
from research_pilot.config import PROVIDERS, load_settings
from research_pilot.tools import Toolbox

from .fixture_server import FixtureServer
from .grade import grade

BENCH_DIR = Path(__file__).parent
PRICES_PER_MTOK = {  # input, output (USD)
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def load_tasks(path: Path, only: set[str] | None) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [t for t in tasks if not only or t["id"] in only]


def load_done(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    return [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    if provider != "anthropic":
        return 0.0  # Groq free tier / local Ollama
    if model not in PRICES_PER_MTOK:
        return None
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


LOCKED_KEYS = ("agent", "provider", "model", "defenses", "tasks_file", "prompts")
LOCK_DEFAULTS = {"prompts": "v1"}  # folders started before a key existed used its default


def check_config(out_dir: Path, config: dict) -> str | None:
    """A results folder keeps the settings it was started with, so a resume can't mix configurations."""
    path = out_dir / "config.json"
    if not path.exists():
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    get = lambda c, k: c.get(k, LOCK_DEFAULTS.get(k))
    diffs = [f"{k}: {get(saved, k)!r} vs now {get(config, k)!r}" for k in LOCKED_KEYS if get(saved, k) != get(config, k)]
    return "; ".join(diffs) or None


def summarize(rows: list[dict], provider: str, model: str) -> dict:
    n = len(rows)
    by_category = defaultdict(list)
    by_task = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r["passed"])
        by_task[r["task_id"]].append(r["passed"])
    tool_calls = sum(r["tool_calls"] for r in rows)
    blocked = sum(r.get("tool_calls_blocked", 0) for r in rows)
    adversarial = [r for r in rows if r["category"] == "adversarial"]
    input_tokens = sum(r["input_tokens"] for r in rows)
    output_tokens = sum(r["output_tokens"] for r in rows)
    return {
        "runs": n,
        "task_success": sum(r["passed"] for r in rows) / n,
        "by_category": {c: sum(v) / len(v) for c, v in sorted(by_category.items())},
        "by_task": {t: f"{sum(v)}/{len(v)}" for t, v in by_task.items()},
        "tool_calls": tool_calls,
        "tool_success": sum(r["tool_calls_ok"] for r in rows) / tool_calls if tool_calls else None,
        "tool_calls_blocked": blocked,
        "injection_followed": f"{sum(r['injection_followed'] for r in adversarial)}/{len(adversarial)}" if adversarial else None,
        "avg_llm_calls": sum(r["steps"] for r in rows) / n,
        "avg_tool_calls": tool_calls / n,
        "avg_wall_time_s": sum(r["wall_time_s"] for r in rows) / n,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "est_cost_usd": cost_usd(provider, model, input_tokens, output_tokens),
    }


def summary_markdown(name: str, config: dict, s: dict) -> str:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    cost = "n/a" if s["est_cost_usd"] is None else f"${s['est_cost_usd']:.2f}"
    lines = [
        f"# Benchmark run: {name}",
        "",
        f"agent `{config['agent']}` | {config['provider']} `{config['model']}` | defenses `{'on' if config['defenses'] else 'off'}` "
        f"| prompts `{config.get('prompts', 'v1')}` | tasks `{config['tasks_file']}` | {s['runs']} runs",
        "",
        "| metric | value |",
        "|---|---|",
        f"| task success | {pct(s['task_success'])} |",
        f"| tool success | {pct(s['tool_success'])} ({s['tool_calls']} calls) |",
        f"| tool calls blocked by policy | {s['tool_calls_blocked']} |",
        f"| injection followed | {s['injection_followed']} |",
        f"| avg LLM calls / run | {s['avg_llm_calls']:.1f} |",
        f"| avg tool calls / run | {s['avg_tool_calls']:.1f} |",
        f"| avg wall time / run | {s['avg_wall_time_s']:.1f}s |",
        f"| tokens in / out | {s['input_tokens']:,} / {s['output_tokens']:,} |",
        f"| est. cost | {cost} |",
        f"| stop reasons | {s['stop_reasons']} |",
        "",
        "| category | success |",
        "|---|---|",
        *[f"| {c} | {pct(v)} |" for c, v in s["by_category"].items()],
        "",
        "| task | passed |",
        "|---|---|",
        *[f"| {t} | {v} |" for t, v in s["by_task"].items()],
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which can't print many characters models emit.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the ResearchPilot benchmark.")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="plan")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks", help="comma-separated task ids (default: all)")
    parser.add_argument("--name", help="results folder name (default: <agent>_<model>_<date>)")
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--model")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--port", type=int, default=8765, help="fixture server port (tasks reference it)")
    parser.add_argument("--defenses", choices=["on", "off"], help="week 3 hardening (default: RP_DEFENSES, on)")
    parser.add_argument("--tasks-file", default="tasks.jsonl", help="task file inside bench/")
    parser.add_argument("--prompts", choices=["v1", "v2"], help="plan agent prompt version (default: RP_PROMPTS, v1)")
    args = parser.parse_args(argv)

    overrides = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.effort:
        overrides["effort"] = args.effort
    if args.defenses:
        overrides["defenses"] = args.defenses == "on"
    if args.prompts:
        overrides["prompt_version"] = args.prompts
    settings = load_settings(**overrides)
    name = args.name or f"{args.agent}_{settings.model.replace('/', '-').replace(':', '-')}_{datetime.now():%Y%m%d}"
    out_dir = BENCH_DIR / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fixture pages are served from 127.0.0.1, so the hosted-deployment SSRF block stays off here.
    settings = load_settings(**overrides, trace_dir=out_dir / "traces", block_private_urls=False)
    config = {"agent": args.agent, "provider": settings.provider, "model": settings.model, "effort": settings.effort,
              "max_steps": settings.max_steps, "max_step_turns": settings.max_step_turns,
              "max_tool_turns": settings.max_tool_turns, "max_run_tokens": settings.max_run_tokens,
              "defenses": settings.defenses, "tasks_file": args.tasks_file, "prompts": settings.prompt_version}
    mismatch = check_config(out_dir, config)
    if mismatch:
        print(f"{name} was started with different settings ({mismatch}). "
              f"Pass the same flags to resume it, or use a new --name.")
        return 2

    tasks = load_tasks(BENCH_DIR / args.tasks_file, set(args.tasks.split(",")) if args.tasks else None)
    results_path = out_dir / "results.jsonl"
    rows = load_done(results_path)
    done = {(r["task_id"], r["repeat"]) for r in rows}
    todo = [(t, k) for k in range(args.repeats) for t in tasks if (t["id"], k) not in done]
    print(f"{name}: {len(tasks)} tasks x {args.repeats} repeats, {len(done)} already done, {len(todo)} to run")

    agent = AGENTS[args.agent](settings, Toolbox(settings))
    with FixtureServer(args.port) as fixtures, results_path.open("a", encoding="utf-8") as out:
        for i, (task, repeat) in enumerate(todo, 1):
            question = task["question"].replace("{fixtures}", fixtures.base_url)
            result = agent.run(question)
            trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
            if result.stop_reason == "api_error":
                # Usually a rate limit (e.g. the free tier's daily cap). Don't record a failure;
                # stop so the same --name can resume later.
                error = next(e["text"] for e in reversed(trace["events"]) if e["type"] == "llm_call")
                print(f"\nStopped at {task['id']} r{repeat}: provider error: {error}")
                print(f"Re-run with --name {name} later to resume.")
                break
            verdict = grade(task, result.answer, trace)
            row = {
                "task_id": task["id"], "category": task["category"], "repeat": repeat,
                **verdict, "answer": result.answer, "plan": result.plan,
                **{k: trace["result"][k] for k in ("stop_reason", "steps", "wall_time_s", "input_tokens",
                                                   "output_tokens", "tool_calls", "tool_calls_ok")},
                "tool_calls_blocked": trace["result"].get("tool_calls_blocked", 0),
                "trace": str(result.trace_path.relative_to(out_dir)),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            mark = "PASS" if verdict["passed"] else "FAIL " + "; ".join(verdict["failures"])
            print(f"[{i}/{len(todo)}] {task['id']} r{repeat}: {mark} "
                  f"({row['tool_calls']} tools, {row['wall_time_s']:.0f}s, {row['stop_reason']})")
        if fixtures.collect_hits:
            print(f"WARNING: exfiltration URL was hit {len(fixtures.collect_hits)} time(s)")

    if not rows:
        return 1
    summary = summarize(rows, settings.provider, settings.model)
    (out_dir / "summary.json").write_text(json.dumps({"config": config, **summary}, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(name, config, summary), encoding="utf-8")
    print("\n" + summary_markdown(name, config, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
