#!/usr/bin/env python
"""Collect every evaluated EoH run into the ATSP benchmark tables.

Reads ``runs/llm/EoH/<task>/<run>/eval_*.json`` and writes a Markdown report
with (a) mean +/- std test gap per task and method, and (b) per-instance
EoH-vs-baseline comparisons.

    python python_scripts/llm/EoH/generate_eoh_tables.py
    python python_scripts/llm/EoH/generate_eoh_tables.py --task gls --out paper/eoh.md
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))

from solvers.llm.EoH.atsp.config import repo_root  # noqa: E402
from solvers.llm.EoH.atsp.registry import TASK_SUMMARY  # noqa: E402

from evaluation.llm.EoH.stats_analysis import (  # noqa: E402
    aggregate_by_task,
    collect,
    per_instance_comparison,
    summary_table,
)

ROOT = repo_root()
DEFAULT_RUNS_ROOT = os.path.join("runs", "llm", "EoH")


def build_report(runs_root: str, task: str | None = None) -> tuple[str, int]:
    rows = []
    for filename in ("eval_test.json", "eval_test_baseline.json"):
        rows.extend(collect(runs_root, task, filename))
    if not rows:
        return "", 0

    aggregated = aggregate_by_task(rows)
    lines = [
        "# EoH on ATSP — benchmark results",
        "",
        f"_generated {datetime.now().isoformat(timespec='seconds')} from "
        f"{len(rows)} evaluated run(s)_",
        "",
        "Test set: the TSPLIB ATSP instances in `data/raw/atsp`; the gap is measured "
        "against the proven optima in `bestSolutions.txt`. Heuristics were evolved on "
        "synthetic instances only, so nothing here was seen during the search.",
        "",
        "## Summary",
        "",
        summary_table(aggregated),
        "",
    ]
    for name in sorted(aggregated):
        lines += [f"## {name} — {TASK_SUMMARY.get(name, '')}", "",
                  per_instance_comparison(rows, name), ""]
    return "\n".join(lines), len(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--task", choices=sorted(TASK_SUMMARY))
    parser.add_argument("--out", default=os.path.join(DEFAULT_RUNS_ROOT,
                                                      "benchmark_tables.md"))
    args = parser.parse_args(argv)

    runs_root = (args.runs_root if os.path.isabs(args.runs_root)
                 else os.path.join(ROOT, args.runs_root))
    report, n_rows = build_report(runs_root, args.task)

    if not n_rows:
        print(f"No eval_*.json found under {runs_root}.\n"
              f"Run python_scripts/llm/EoH/eval_eoh_atsp.py first.")
        return 1

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(report)
    print(f"\nWritten to {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
