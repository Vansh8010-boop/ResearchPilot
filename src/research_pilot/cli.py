"""Command line entry point: `rp "your question"`."""

import argparse
import json
import sys

from .agent import AGENTS
from .config import PROVIDERS, load_settings
from .tools import Toolbox


def _print_event(kind: str, data: dict) -> None:
    if kind == "plan":
        print("Plan:")
        for i, goal in enumerate(data["plan"], 1):
            print(f"  {i}. {goal}")
    elif kind == "finding":
        print(f"\n[step {data['step']} result] {data['finding'][:400]}")
    elif kind == "llm" and data["stop_reason"] == "tool_use" and data["text"]:
        print(f"\n[{data['phase']}] {data['text']}")
    elif kind == "tool":
        status = "ok" if data["ok"] else "ERROR"
        cached = " (cached)" if data["cached"] else ""
        print(f"  -> {data['name']}({json.dumps(data['input'], ensure_ascii=False)}) {status}{cached}")
        if not data["ok"]:
            print(f"     {data['output'][:200]}")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which can't print many characters models emit.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="rp", description="Ask ResearchPilot a research question.")
    parser.add_argument("question")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="plan")
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--model")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--defenses", choices=["on", "off"], help="week 3 hardening (default: on)")
    parser.add_argument("--no-cache", action="store_true", help="ignore and don't write the tool cache")
    parser.add_argument("--quiet", action="store_true", help="print only the final answer")
    args = parser.parse_args(argv)

    overrides = {"use_cache": not args.no_cache}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.effort:
        overrides["effort"] = args.effort
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.defenses:
        overrides["defenses"] = args.defenses == "on"
    settings = load_settings(**overrides)

    agent = AGENTS[args.agent](settings, Toolbox(settings), on_event=None if args.quiet else _print_event)
    result = agent.run(args.question)

    print("\n" + "=" * 70)
    print(result.answer or "(no answer)")
    print("=" * 70)
    print(f"stop: {result.stop_reason} | steps: {result.steps} | trace: {result.trace_path}")
    return 0 if result.stop_reason in ("answered", "max_steps") else 1


if __name__ == "__main__":
    sys.exit(main())
