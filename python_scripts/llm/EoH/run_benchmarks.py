#!/usr/bin/env python
"""Score every EoH heuristic on the held-out TSPLIB ATSP test set.

Walks `runs/llm/EoH/<task>/*/best_heuristic.py` and evaluates each one, plus
the hand-crafted baseline of every task, so the comparison tables have both
sides. Delegates to eval_eoh_atsp.py, one subprocess per evaluation, so a
broken heuristic can never abort the sweep.

    python python_scripts/llm/EoH/run_benchmarks.py                # baselines + all runs
    python python_scripts/llm/EoH/run_benchmarks.py --baselines-only
    python python_scripts/llm/EoH/run_benchmarks.py --runs-only    # skip baselines you already have
    python python_scripts/llm/EoH/run_benchmarks.py --task gls --max-n 100
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))

from solvers.llm.EoH.atsp.config import repo_root  # noqa: E402
from solvers.llm.EoH.atsp.registry import TASK_SUMMARY  # noqa: E402

ROOT = repo_root()
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval_eoh_atsp.py")
RUNS = os.path.join(ROOT, "runs", "llm", "EoH")


def _run(cmd: list[str]) -> int:
    printable = [os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd]
    print("\n$ " + " ".join(printable))
    return subprocess.call(cmd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASK_SUMMARY), action="append",
                        help="restrict to these tasks (repeatable)")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--runs-only", action="store_true",
                        help="skip the baselines (useful once they are scored)")
    parser.add_argument("--max-n", type=int, help="skip instances larger than this")
    args = parser.parse_args(argv)

    tasks = args.task or sorted(TASK_SUMMARY)
    extra = ["--max-n", str(args.max_n)] if args.max_n else []
    failures = 0

    for task in tasks:
        if not args.runs_only:
            failures += bool(_run([sys.executable, EVAL, "--task", task,
                                   "--baseline", *extra]))
        if args.baselines_only:
            continue

        found = sorted(glob.glob(os.path.join(RUNS, task, "*", "best_heuristic.py")))
        if not found:
            print(f"  (no finished EoH runs for task '{task}')")
        for path in found:
            failures += bool(_run([sys.executable, EVAL,
                                   "--run", os.path.dirname(path), *extra]))

    print("\nNow build the tables:")
    print("  python python_scripts/llm/EoH/generate_paper_tables.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
