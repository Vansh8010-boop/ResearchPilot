"""Copy a benchmark run's trace into demo/ so the UI can replay it as a recorded run.

    uv run python -m bench.export_demo --name plan_baseline --task a2 --repeat 0 --id exfil-defenses-off \\
        --title "Injection: silent data leak (defenses off)" --description "..."

Re-exporting the same --id replaces that entry.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from .run import BENCH_DIR, load_done

DEMO_DIR = BENCH_DIR.parent / "public" / "demo"


def export(name: str, task_id: str, repeat: int, run_id: str, title: str, description: str,
           demo_dir: Path = DEMO_DIR, bench_dir: Path = BENCH_DIR) -> Path:
    out_dir = bench_dir / "results" / name
    row = next((r for r in load_done(out_dir / "results.jsonl") if r["task_id"] == task_id and r["repeat"] == repeat),
               None)
    if row is None:
        raise SystemExit(f"no run {task_id} r{repeat} in {name}")

    (demo_dir / "runs").mkdir(parents=True, exist_ok=True)
    target = demo_dir / "runs" / f"{run_id}.json"
    shutil.copyfile(out_dir / row["trace"], target)

    index_path = demo_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    entry = {"id": run_id, "file": f"runs/{run_id}.json", "title": title, "description": description,
             "source": f"{name} {task_id} r{repeat}"}
    index = [e for e in index if e["id"] != run_id] + [entry]
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Export a benchmark trace as a recorded demo run.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--id", required=True, help="short slug for the demo entry")
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args(argv)
    target = export(args.name, args.task, args.repeat, args.id, args.title, args.description)
    print(f"exported to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
