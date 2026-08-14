#!/usr/bin/env python
"""Collect every evaluated MCTS-AHD run into the ATSP benchmark tables.

Reads `runs/llm/MCTS-AHD/<task>/<run>/eval_*.json` and writes a Markdown report
with (a) mean +/- std test gap per task and method and (b) per-instance
comparisons against the seed heuristic.

    python python_scripts/llm/MCTS-AHD/generate_paper_tables.py
    python python_scripts/llm/MCTS-AHD/generate_paper_tables.py --task atsp_kgls

Point it at the whole `runs/llm` tree to put EoH, ReEvo and MCTS-AHD in one
table — the result schema is shared and `stats_analysis` keys rows by the
`framework` field, not by the path:

    python python_scripts/llm/MCTS-AHD/generate_paper_tables.py \\
        --runs-root runs/llm --out runs/benchmark_tables.md
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.llm.EoH.stats_analysis import (  # noqa: E402
    aggregate_by_task,
    collect,
    per_instance_comparison,
    summary_table,
)
from evaluation.llm.mcts_ahd import TASK_SUMMARY  # noqa: E402

DEFAULT_RUNS_ROOT = os.path.join("runs", "llm", "MCTS-AHD")


def build_report(runs_root: str, task: str | None = None) -> tuple[str, int]:
    rows = []
    # eval_test          — a designed heuristic, one per finished run
    # eval_test_seed     — the task's seed function, the baseline
    # eval_test_baseline — EoH's hand-written references, if this is the shared tree
    for filename in ("eval_test.json", "eval_test_seed.json",
                     "eval_test_baseline.json"):
        rows.extend(collect(runs_root, task, filename))
    if not rows:
        return "", 0

    aggregated = aggregate_by_task(rows)
    lines = [
        "# MCTS-AHD on ATSP — benchmark results",
        "",
        f"_generated {datetime.now().isoformat(timespec='seconds')} from "
        f"{len(rows)} evaluated run(s)_",
        "",
        "Test set: the TSPLIB ATSP instances in `data/raw/atsp`; the gap is "
        "measured against the proven optima in `bestSolutions.txt`. Under the "
        "default data config two of those instances (rbg323, rbg403) are also "
        "in the training split — see the `train_overlap` field of each "
        "`eval_test.json`, and re-run with `--exclude-train` for the held-out "
        "number.",
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
              f"Run python_scripts/llm/MCTS-AHD/run_benchmarks.py first.")
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
