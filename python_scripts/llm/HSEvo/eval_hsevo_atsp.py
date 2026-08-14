#!/usr/bin/env python
"""Benchmark an HSEvo-designed heuristic on the held-out TSPLIB ATSP instances.

    # Evaluate every finished run under runs/llm/HSEvo/
    python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --all

    # Evaluate one specific run
    python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --run runs/llm/HSEvo/atsp_gls-gls/<date>_<time>

    # Evaluate HSEvo baseline seed
    python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --task atsp_gls --seed-heuristic
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HSEVO_SOLVER = os.path.join(ROOT, "solvers", "llm", "HSEvo")
for _p in (ROOT, HSEVO_SOLVER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.llm.HSEvo.benchmark_runner import (
    TASKS,
    load_heuristic_for_run,
    run_benchmark,
    seed_heuristic,
    write_results,
)
from evaluation.metrics import markdown_table
from atsp.data import resolve_split

RUNS = os.path.join(ROOT, "runs", "llm", "HSEvo")


def evaluate(task: str, code: str, label: str, out_dir: str, name: str,
             max_n: int | None, names: list[str] | None) -> dict:
    spec = {"source": "tsplib", "dir": "data/raw/atsp",
            "best_known": "data/raw/atsp/bestSolutions.txt"}
    if max_n:
        spec["max_n"] = max_n
    if names:
        spec["names"] = names
    instances = resolve_split(spec, ROOT, log=lambda *a: None)

    print(f"\n=== {task}: {label} on {len(instances)} TSPLIB instances ===")
    records = run_benchmark(task, code, instances)
    written = write_results(out_dir, name, records, {
        "title": f"ATSP {task} — {label}", "framework": "HSEvo",
        "task": task, "heuristic": label, "split": "test",
    })
    print(markdown_table(records))
    summary = written["summary"]
    print(f"\n  mean gap : {summary['mean_gap_percent']:.3f}%  "
          f"(median {summary['median_gap_percent']:.3f}%)")
    print(f"  solved   : {summary['n_solved']}/{summary['n_instances']}, "
          f"{summary['n_optimal']} at the optimum")
    print(f"  written  : {os.path.relpath(written['csv'], ROOT)}")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="an HSEvo run directory")
    parser.add_argument("--all", action="store_true",
                        help="every finished run under runs/llm/HSEvo/")
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--heuristic", help="path to a heuristic .py file")
    parser.add_argument("--seed-heuristic", action="store_true",
                        help="score HSEvo's baseline seed function")
    parser.add_argument("--max-n", type=int)
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--name", help="basename of result files")
    args = parser.parse_args(argv)

    if args.all:
        found = sorted(glob.glob(os.path.join(RUNS, "*", "*", "best_heuristic.py")))
        if not found:
            print(f"No finished runs under {os.path.relpath(RUNS, ROOT)}.")
            return 1
        for path in found:
            run_dir = os.path.dirname(path)
            task, code = load_heuristic_for_run(run_dir)
            evaluate(task, code, "best_heuristic.py", args.out or run_dir,
                     args.name or "eval_test", args.max_n, args.names)
        return 0

    if args.run:
        run_dir = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
        task, code = load_heuristic_for_run(run_dir)
        evaluate(task, code, "best_heuristic.py", args.out or run_dir,
                 args.name or "eval_test", args.max_n, args.names)
        return 0

    if not args.task:
        parser.error("provide --run, --all, or --task")

    if args.seed_heuristic or not args.heuristic:
        code, label = seed_heuristic(args.task), "baseline: HSEvo seed function"
        name = "eval_test_seed"
    else:
        code = open(args.heuristic, encoding="utf-8").read()
        label, name = os.path.basename(args.heuristic), "eval_test"
    out_dir = args.out or os.path.join(RUNS, "eval", args.task)
    evaluate(args.task, code, label, out_dir, args.name or name,
             args.max_n, args.names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
