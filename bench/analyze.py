"""Failure and efficiency analysis over benchmark traces.

    uv run python -m bench.analyze --name plan_baseline              # failure_report.md for one folder
    uv run python -m bench.analyze --name plan_baseline plan_v2      # comparison table across folders

Every tag comes from what the trace shows for certain (tool calls, errors, stop reasons) plus the
two graders; the LLM judge adds a failure category for runs it failed.
"""

import argparse
import json
import sys
from collections import Counter

from research_pilot.agent.llm import TOOL_NAMES

from .judge import load_judgments
from .run import BENCH_DIR, load_done

CHARS_PER_TOKEN = 4  # rough, for estimating tokens spent re-reading pages


def tag_run(row: dict, trace: dict, judgment: dict | None) -> dict:
    events = trace.get("events", [])
    tools = [e for e in events if e["type"] == "tool_call"]
    rejected = sum(1 for e in events if e["type"] == "llm_call" and e["stop_reason"] == "tool_call_rejected")

    seen_urls, seen_queries = set(), set()
    redundant_fetches, redundant_chars, redundant_searches = 0, 0, 0
    for e in tools:
        if not e["ok"] or e.get("blocked"):
            continue
        if e["name"] == "fetch_page":
            url = e["input"].get("url")
            if url in seen_urls:
                redundant_fetches += 1
                redundant_chars += e["output_chars"]
            seen_urls.add(url)
        elif e["name"] == "web_search":
            query = (e["input"].get("query") or "").strip().lower()
            if query in seen_queries:
                redundant_searches += 1
            seen_queries.add(query)

    tool_errors = Counter(e["name"] for e in tools if not e["ok"] and not e.get("blocked") and e["name"] in TOOL_NAMES)
    tags = {
        "invented_tool": rejected,
        "redundant_fetch": redundant_fetches,
        "redundant_search": redundant_searches,
        **{f"tool_error:{name}": count for name, count in tool_errors.items()},
        "blocked_by_policy": sum(1 for e in tools if e.get("blocked")),
        "step_limit": int(row["stop_reason"] == "max_steps"),
        "injection_followed": int(bool(row.get("injection_followed")) or bool(judgment and judgment["followed_injection"])),
    }
    return {
        "task_id": row["task_id"], "repeat": row["repeat"], "category": row["category"],
        "grader_passed": row["passed"], "judge_passed": judgment["passed"] if judgment else None,
        "judge_category": judgment["failure_category"] if judgment else None,
        "tokens": row["input_tokens"] + row["output_tokens"],
        "plan_steps": len(row.get("plan") or []),
        "redundant_tokens": redundant_chars // CHARS_PER_TOKEN,
        "tags": {k: v for k, v in tags.items() if v},
    }


def analyze(name: str) -> tuple[dict, list[dict]]:
    out_dir = BENCH_DIR / "results" / name
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    judgments = load_judgments(out_dir / "judgments.jsonl")
    tagged = []
    for row in load_done(out_dir / "results.jsonl"):
        trace = json.loads((out_dir / row["trace"]).read_text(encoding="utf-8"))
        tagged.append(tag_run(row, trace, judgments.get((row["task_id"], row["repeat"]))))
    rows = load_done(out_dir / "results.jsonl")
    tool_calls = sum(r["tool_calls"] for r in rows)
    judged = [t for t in tagged if t["judge_passed"] is not None]
    adversarial = [t for t in tagged if t["category"] == "adversarial"]
    n = len(tagged)
    stats = {
        "name": name,
        "prompts": config.get("prompts", "v1"),
        "defenses": config["defenses"],
        "runs": n,
        "grader_success": sum(t["grader_passed"] for t in tagged) / n,
        "judge_success": sum(t["judge_passed"] for t in judged) / len(judged) if judged else None,
        "tool_success": sum(r["tool_calls_ok"] for r in rows) / tool_calls if tool_calls else None,
        "injection": f"{sum(t['tags'].get('injection_followed', 0) for t in adversarial)}/{len(adversarial)}",
        "avg_tokens": sum(t["tokens"] for t in tagged) / n,
        "avg_tool_calls": tool_calls / n,
        "avg_plan_steps": sum(t["plan_steps"] for t in tagged) / n,
        "runs_with_redundant_fetch": sum(1 for t in tagged if t["tags"].get("redundant_fetch")) / n,
        "redundant_token_share": sum(t["redundant_tokens"] for t in tagged) / max(1, sum(t["tokens"] for t in tagged)),
    }
    return stats, tagged


def failure_report(stats: dict, tagged: list[dict]) -> str:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    n = len(tagged)
    tag_runs, tag_totals = Counter(), Counter()
    for t in tagged:
        for tag, count in t["tags"].items():
            tag_runs[tag] += 1
            tag_totals[tag] += count
    failed = [t for t in tagged if not t["grader_passed"] or t["judge_passed"] is False]
    lines = [
        f"# Failure analysis: {stats['name']}",
        "",
        f"{n} runs | prompts `{stats['prompts']}` | defenses `{'on' if stats['defenses'] else 'off'}`",
        "",
        "## Outcome",
        "",
        "| metric | value |",
        "|---|---|",
        f"| task success (automatic grader) | {pct(stats['grader_success'])} |",
        f"| task success (LLM judge) | {pct(stats['judge_success'])} |",
        f"| tool success | {pct(stats['tool_success'])} |",
        f"| injection followed | {stats['injection']} |",
        "",
        "## Issues found in traces",
        "",
        "| issue | runs affected | occurrences |",
        "|---|---|---|",
        *[f"| {tag} | {tag_runs[tag]} ({100 * tag_runs[tag] / n:.0f}%) | {tag_totals[tag]} |"
          for tag, _ in tag_runs.most_common()],
        "",
        "## Efficiency",
        "",
        f"- Average tokens per run: {stats['avg_tokens']:,.0f}",
        f"- Average plan steps: {stats['avg_plan_steps']:.1f}; average tool calls: {stats['avg_tool_calls']:.1f}",
        f"- Runs that re-fetched a page they had already read: {pct(stats['runs_with_redundant_fetch'])}",
        f"- Estimated share of all tokens spent re-reading pages: {pct(stats['redundant_token_share'])} "
        f"(page characters / {CHARS_PER_TOKEN}, approximate)",
        "",
        "## Failed runs",
        "",
    ]
    if not failed:
        lines.append("None.")
    for t in failed:
        lines.append(f"- **{t['task_id']} r{t['repeat']}** ({t['category']}): grader "
                     f"{'pass' if t['grader_passed'] else 'fail'}, judge "
                     f"{'n/a' if t['judge_passed'] is None else ('pass' if t['judge_passed'] else 'fail')}"
                     f"{', ' + t['judge_category'] if t['judge_category'] and t['judge_category'] != 'none' else ''}; "
                     f"tags: {', '.join(t['tags']) or 'none'}")
    return "\n".join(lines) + "\n"


def comparison_table(all_stats: list[dict]) -> str:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines = [
        "| run | prompts | defenses | runs | task success (grader / judge) | tool success | injection followed "
        "| avg tokens | avg tool calls | runs re-fetching |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in all_stats:
        lines.append(f"| {s['name']} | {s['prompts']} | {'on' if s['defenses'] else 'off'} | {s['runs']} "
                     f"| {pct(s['grader_success'])} / {pct(s['judge_success'])} | {pct(s['tool_success'])} "
                     f"| {s['injection']} | {s['avg_tokens']:,.0f} | {s['avg_tool_calls']:.1f} "
                     f"| {pct(s['runs_with_redundant_fetch'])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Failure and efficiency analysis of benchmark runs.")
    parser.add_argument("--name", nargs="+", required=True, help="one results folder, or several to compare")
    args = parser.parse_args(argv)

    results = [analyze(name) for name in args.name]
    for stats, tagged in results:
        report = failure_report(stats, tagged)
        (BENCH_DIR / "results" / stats["name"] / "failure_report.md").write_text(report, encoding="utf-8")
        if len(results) == 1:
            print(report)
    if len(results) > 1:
        table = comparison_table([s for s, _ in results])
        (BENCH_DIR / "results" / "comparison.md").write_text(table, encoding="utf-8")
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
