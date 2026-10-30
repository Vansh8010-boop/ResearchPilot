"""Compare the LLM judge and the automatic grader against hand labels.

    uv run python -m bench.calibrate --name plan_baseline

Reports agreement and Cohen's kappa for each grader against the human labels. Kappa corrects for
agreement by chance: with mostly-passing runs, a grader that says "pass" to everything agrees often
but scores a kappa near 0. Writes calibration.md into the results folder.
"""

import argparse
import json
import sys

from .judge import load_judgments
from .label import load_labels
from .run import BENCH_DIR, load_done


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None  # both raters gave one label to everything: kappa is undefined
    return (observed - expected) / (1 - expected)


def compare(human: list[bool], grader: list[bool]) -> dict:
    tp = sum(h and g for h, g in zip(human, grader))
    tn = sum(not h and not g for h, g in zip(human, grader))
    fp = sum(not h and g for h, g in zip(human, grader))  # grader passed what the human failed
    fn = sum(h and not g for h, g in zip(human, grader))
    n = len(human)
    return {"n": n, "agreement": (tp + tn) / n if n else None, "kappa": cohens_kappa(human, grader),
            "false_pass": fp, "false_fail": fn}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Calibrate graders against hand labels.")
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    out_dir = BENCH_DIR / "results" / args.name
    rows = {(r["task_id"], r["repeat"]): r for r in load_done(out_dir / "results.jsonl")}
    judgments = load_judgments(out_dir / "judgments.jsonl")
    labels = [x for k, x in load_labels().items() if k[0] == args.name]
    if not labels:
        print(f"No hand labels for {args.name} yet. Run: uv run python -m bench.label --name {args.name}")
        return 1

    keys = [(x["task_id"], x["repeat"]) for x in labels if (x["task_id"], x["repeat"]) in judgments]
    human = [next(x["passed"] for x in labels if (x["task_id"], x["repeat"]) == k) for k in keys]
    judge = compare(human, [judgments[k]["passed"] for k in keys])
    grader = compare(human, [rows[k]["passed"] for k in keys])

    fmt = lambda v, pct=False: "n/a" if v is None else (f"{100 * v:.1f}%" if pct else f"{v:.2f}")
    lines = [
        f"# Calibration: {args.name}",
        "",
        f"{len(keys)} hand-labelled runs ({sum(human)} pass, {len(human) - sum(human)} fail by hand).",
        "",
        "| grader | agreement with human | Cohen's kappa | false passes | false fails |",
        "|---|---|---|---|---|",
        f"| LLM judge | {fmt(judge['agreement'], True)} | {fmt(judge['kappa'])} | {judge['false_pass']} | {judge['false_fail']} |",
        f"| automatic grader | {fmt(grader['agreement'], True)} | {fmt(grader['kappa'])} | {grader['false_pass']} | {grader['false_fail']} |",
        "",
        "Kappa: 1 = perfect, 0 = no better than chance. Above about 0.6 is usually treated as substantial agreement.",
        "",
        "## Where a grader disagrees with you",
        "",
    ]
    for (task_id, repeat), h in zip(keys, human):
        j, g = judgments[(task_id, repeat)]["passed"], rows[(task_id, repeat)]["passed"]
        if j != h or g != h:
            note = next(x["note"] for x in labels if (x["task_id"], x["repeat"]) == (task_id, repeat))
            lines.append(f"- **{task_id} r{repeat}**: you {'pass' if h else 'fail'}, judge {'pass' if j else 'fail'}, "
                         f"grader {'pass' if g else 'fail'}. {judgments[(task_id, repeat)]['reason']}"
                         + (f" Your note: {note}" if note else ""))
    text = "\n".join(lines) + "\n"
    (out_dir / "calibration.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
