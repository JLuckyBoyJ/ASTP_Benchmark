#!/usr/bin/env python
"""Benchmark a ReEvo-designed heuristic on the held-out TSPLIB ATSP instances.

Scores heuristics through the *same* problem engines ReEvo evolved them on, and
writes the same `eval_test.{csv,json,md}` files the EoH side produces, so both
frameworks land in one comparison table.

    # every finished run under runs/llm/ReEvo/
    python python_scripts/llm/ReEvo/eval_reevo_atsp.py --all

    # one run, or one heuristic file
    python python_scripts/llm/ReEvo/eval_reevo_atsp.py --run runs/llm/ReEvo/atsp_gls-gls/<date>_<time>
    python python_scripts/llm/ReEvo/eval_reevo_atsp.py --task atsp_gls --seed-heuristic
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT_ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (ROOT_, os.path.join(ROOT_, "solvers", "llm", "ReEvo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.llm.ReEvo.benchmark_runner import (  # noqa: E402
    TASKS,
    load_heuristic_for_run,
    run_benchmark,
    seed_heuristic,
    write_results,
)
from evaluation.metrics import markdown_table  # noqa: E402
from atsp.data import resolve_split  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "llm", "ReEvo")


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
        "title": f"ATSP {task} — {label}", "framework": "ReEvo",
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
    parser.add_argument("--run", help="a ReEvo run directory")
    parser.add_argument("--all", action="store_true",
                        help="every finished run under runs/llm/ReEvo/")
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--heuristic", help="path to a heuristic .py file")
    parser.add_argument("--seed-heuristic", action="store_true",
                        help="score ReEvo's own seed function (the baseline)")
    parser.add_argument("--max-n", type=int)
    parser.add_argument("--names", nargs="*")
    # A partial sweep (--names/--max-n) writes the same filenames as a full one
    # and silently replaces it. Send exploratory runs somewhere else.
    parser.add_argument("--out", help="output directory "
                                      "(default: the run dir, else runs/llm/ReEvo/eval/<task>)")
    parser.add_argument("--name", help="basename of the result files "
                                       "(default: eval_test, or eval_test_seed)")
    args = parser.parse_args(argv)

    if (args.names or args.max_n) and not (args.out or args.name):
        print("note: this is a partial instance set but writes the standard "
              "result files; pass --name to keep it separate from a full sweep.")

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
        # The label must start with "baseline" — that prefix is how
        # evaluation/llm/EoH/stats_analysis.py tells a hand-written reference
        # heuristic from an evolved one when it builds the comparison table.
        code, label = seed_heuristic(args.task), "baseline: ReEvo seed function"
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
