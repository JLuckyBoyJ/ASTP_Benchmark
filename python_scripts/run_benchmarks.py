#!/usr/bin/env python
"""Benchmark every solver family on the TSPLIB ATSP test set.

Repo-level driver. Per-family logic lives next to the family itself:

    llm/EoH  ->  python_scripts/llm/EoH/eval_eoh_atsp.py

For EoH this scores the hand-crafted baseline of each task plus the best
heuristic of every finished run under ``runs/llm/EoH/``. Classical, exact and
neural solvers plug in the same way once implemented.

    python python_scripts/run_benchmarks.py                # baselines + all runs
    python python_scripts/run_benchmarks.py --baselines-only
    python python_scripts/run_benchmarks.py --task gls --max-n 100
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solvers.llm.EoH.atsp.config import repo_root  # noqa: E402
from solvers.llm.EoH.atsp.registry import TASK_SUMMARY  # noqa: E402

ROOT = repo_root()
EOH_EVAL = os.path.join(ROOT, "python_scripts", "llm", "EoH", "eval_eoh_atsp.py")
EOH_RUNS = os.path.join(ROOT, "runs", "llm", "EoH")


def _run(cmd: list[str]) -> int:
    printable = [os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd]
    print("\n$ " + " ".join(printable))
    return subprocess.call(cmd)


def benchmark_eoh(tasks: list[str], baselines: bool, runs: bool,
                  extra: list[str]) -> int:
    failures = 0
    for task in tasks:
        if baselines:
            failures += bool(_run([sys.executable, EOH_EVAL, "--task", task,
                                   "--baseline", *extra]))
        if runs:
            pattern = os.path.join(EOH_RUNS, task, "*", "best_heuristic.py")
            found = sorted(glob.glob(pattern))
            if not found:
                print(f"  (no finished EoH runs for task '{task}')")
            for path in found:
                failures += bool(_run([sys.executable, EOH_EVAL,
                                       "--run", os.path.dirname(path), *extra]))
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASK_SUMMARY), action="append",
                        help="restrict to these EoH tasks (repeatable)")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--runs-only", action="store_true")
    parser.add_argument("--max-n", type=int, help="skip instances larger than this")
    args = parser.parse_args(argv)

    tasks = args.task or sorted(TASK_SUMMARY)
    extra = ["--max-n", str(args.max_n)] if args.max_n else []

    failures = benchmark_eoh(tasks,
                             baselines=not args.runs_only,
                             runs=not args.baselines_only,
                             extra=extra)

    print("\nNow build the tables:")
    print("  python python_scripts/generate_paper_tables.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
