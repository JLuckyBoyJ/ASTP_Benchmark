#!/usr/bin/env python
"""Score every MoH heuristic on the held-out ATSP test set.

Walks `runs/llm/MoH/<task>-<type>/*/best_heuristic.py`, evaluates each one plus
each task's seed heuristic, and delegates to eval_moh_atsp.py — one subprocess
per evaluation, so a heuristic that hangs or crashes can never abort the sweep.

    python python_scripts/llm/MoH/run_benchmarks.py                # seeds + all runs
    python python_scripts/llm/MoH/run_benchmarks.py --baselines-only
    python python_scripts/llm/MoH/run_benchmarks.py --runs-only
    python python_scripts/llm/MoH/run_benchmarks.py --every-subtask
    python python_scripts/llm/MoH/run_benchmarks.py --task atsp_gls --max-n 100
    python python_scripts/llm/MoH/run_benchmarks.py --exclude-train --name eval_heldout
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "solvers", "llm", "MoH")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.llm.MoH import TASKS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval_moh_atsp.py")
RUNS = os.path.join(ROOT, "runs", "llm", "MoH")


def _run(cmd: list[str]) -> int:
    printable = [os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd]
    print("\n$ " + " ".join(printable))
    return subprocess.call(cmd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), action="append",
                        help="restrict to these tasks (repeatable)")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--runs-only", action="store_true",
                        help="skip the seed heuristics (once they are scored)")
    parser.add_argument("--every-subtask", action="store_true",
                        help="score each subtask's heuristic, not just the headline")
    parser.add_argument("--max-n", type=int, help="skip instances larger than this")
    parser.add_argument("--exclude-train", action="store_true",
                        help="drop the instances the search was scored on")
    parser.add_argument("--name", help="basename of the result files")
    args = parser.parse_args(argv)

    tasks = args.task or sorted(TASKS)
    extra: list[str] = []
    if args.max_n:
        extra += ["--max-n", str(args.max_n)]
    if args.exclude_train:
        extra += ["--exclude-train"]
    if args.name:
        extra += ["--name", args.name]
    failures = 0

    for task in tasks:
        if not args.runs_only:
            failures += bool(_run([sys.executable, EVAL, "--task", task,
                                   "--seed-heuristic", *extra]))
        if args.baselines_only:
            continue

        found = sorted(glob.glob(os.path.join(RUNS, f"{task}-*", "*",
                                              "best_heuristic.py")))
        if not found:
            print(f"  (no finished MoH runs for task '{task}')")
        for path in found:
            cmd = [sys.executable, EVAL, "--run", os.path.dirname(path), *extra]
            if args.every_subtask:
                cmd.append("--every-subtask")
            failures += bool(_run(cmd))

    print("\nNow build the tables:")
    print("  python python_scripts/llm/MoH/generate_paper_tables.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
