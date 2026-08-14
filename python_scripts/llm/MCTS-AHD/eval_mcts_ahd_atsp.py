#!/usr/bin/env python
"""Benchmark an MCTS-AHD-designed heuristic on the held-out TSPLIB ATSP instances.

Scores heuristics through the *same* engines MCTS-AHD designed them on, and
writes the same `eval_test.{csv,json,md}` files the EoH and ReEvo sides produce,
so all three frameworks land in one comparison table.

    # every finished run under runs/llm/MCTS-AHD/
    python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --all

    # one run, one heuristic file, or the task's seed function (the baseline)
    python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --run runs/llm/MCTS-AHD/atsp_gls-gls/<date>_<time>
    python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_kgls --heuristic path/to/heuristic.py
    python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_gls --seed-heuristic

Which instances are scored, and which were trained on, both come from the active
data config (`configs/llm/MCTS-AHD/cfg/data/<ATSP_DATA>.yaml`, default `tsplib`).
Under that default two of the nineteen TSPLIB instances (rbg323, rbg403) are in
the training split, so `--exclude-train` is offered for the generalisation
number. The full 19-instance mean stays the headline figure for comparability
with the other frameworks, which train on the same pair.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# `evaluation/llm/MCTS-AHD/` has a hyphen in its name and cannot be imported
# directly; `evaluation.llm.mcts_ahd` is a shim that loads it by path.
from evaluation.llm.mcts_ahd import (  # noqa: E402
    TASKS, load_heuristic_for_run, run_benchmark, seed_heuristic, write_results,
)
from evaluation.metrics import markdown_table  # noqa: E402
from atsp.data import resolve_split  # noqa: E402
from atsp_utils import data_config_name, test_split, trained_on  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "llm", "MCTS-AHD")


def evaluate(task: str, code: str, label: str, out_dir: str, name: str,
             max_n: int | None, names: list[str] | None,
             exclude_train: bool) -> dict:
    # The reported set is the `test` split of the active data config, so
    # changing configs/llm/MCTS-AHD/cfg/data/<name>.yaml changes what is
    # reported as well as what is evolved on — they cannot drift apart.
    spec = dict(test_split())
    if max_n:
        spec["max_n"] = max_n
    if names:
        spec["names"] = names
    instances = resolve_split(spec, ROOT, log=lambda *a: None)

    trained = trained_on()
    if exclude_train and trained:
        instances = [i for i in instances if i.name not in trained]

    overlap = sorted(trained.intersection(i.name for i in instances))
    print(f"\n=== {task}: {label} on {len(instances)} TSPLIB instances ===")
    if overlap:
        print(f"    note: {', '.join(overlap)} are in the training split; "
              f"pass --exclude-train for the held-out number")

    records = run_benchmark(task, code, instances)
    written = write_results(out_dir, name, records, {
        "title": f"ATSP {task} — {label}", "framework": "MCTS-AHD",
        "task": task, "heuristic": label, "split": "test",
        "data_config": data_config_name(), "train_overlap": overlap,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="an MCTS-AHD run directory")
    parser.add_argument("--all", action="store_true",
                        help="every finished run under runs/llm/MCTS-AHD/")
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--heuristic", help="path to a heuristic .py file")
    parser.add_argument("--seed-heuristic", action="store_true",
                        help="score the task's seed function (the baseline)")
    parser.add_argument("--exclude-train", action="store_true",
                        help="drop the instances the search trained on")
    parser.add_argument("--max-n", type=int)
    parser.add_argument("--names", nargs="*")
    # A partial sweep (--names/--max-n) writes the same filenames as a full one
    # and would silently replace it. Send exploratory runs somewhere else.
    parser.add_argument("--out", help="output directory (default: the run dir, "
                                      "else runs/llm/MCTS-AHD/eval/<task>)")
    parser.add_argument("--name", help="basename of the result files "
                                       "(default: eval_test, or eval_test_seed)")
    args = parser.parse_args(argv)

    if (args.names or args.max_n or args.exclude_train) and not (args.out or args.name):
        print("note: this is a partial instance set but writes the standard result "
              "files; pass --name to keep it separate from a full sweep.")

    if args.all:
        found = sorted(glob.glob(os.path.join(RUNS, "*", "*", "best_heuristic.py")))
        if not found:
            print(f"No finished runs under {os.path.relpath(RUNS, ROOT)}.")
            return 1
        for path in found:
            run_dir = os.path.dirname(path)
            task, code = load_heuristic_for_run(run_dir)
            evaluate(task, code, "best_heuristic.py", args.out or run_dir,
                     args.name or "eval_test", args.max_n, args.names,
                     args.exclude_train)
        return 0

    if args.run:
        run_dir = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
        task, code = load_heuristic_for_run(run_dir)
        evaluate(task, code, "best_heuristic.py", args.out or run_dir,
                 args.name or "eval_test", args.max_n, args.names, args.exclude_train)
        return 0

    if not args.task:
        parser.error("provide --run, --all, or --task")

    if args.seed_heuristic or not args.heuristic:
        # The label must start with "baseline" — that prefix is how
        # evaluation/llm/EoH/stats_analysis.py tells a hand-written reference
        # heuristic from a designed one when it builds the comparison table.
        code, label = seed_heuristic(args.task), "baseline: MCTS-AHD seed function"
        name = "eval_test_seed"
    else:
        code = open(args.heuristic, encoding="utf-8").read()
        label, name = os.path.basename(args.heuristic), "eval_test"
    out_dir = args.out or os.path.join(RUNS, "eval", args.task)
    evaluate(args.task, code, label, out_dir, args.name or name,
             args.max_n, args.names, args.exclude_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
