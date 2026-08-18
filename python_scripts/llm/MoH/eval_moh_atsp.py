#!/usr/bin/env python
"""Benchmark a MoH-designed heuristic on the held-out TSPLIB ATSP instances.

Scores heuristics through the *same* engines MoH designed them on, and writes
the same `eval_test.{csv,json,md}` files every other framework produces, so all
of them land in one comparison table.

    # every finished run under runs/llm/MoH/
    python python_scripts/llm/MoH/eval_moh_atsp.py --all

    # one run: its headline heuristic, or one subtask's
    python python_scripts/llm/MoH/eval_moh_atsp.py --run runs/llm/MoH/atsp_gls-gls/<date>_<time>
    python python_scripts/llm/MoH/eval_moh_atsp.py --run <run> --subtask atsp_gls-50
    python python_scripts/llm/MoH/eval_moh_atsp.py --run <run> --every-subtask

    # a file, or the task's seed function (the baseline)
    python python_scripts/llm/MoH/eval_moh_atsp.py --task atsp_kgls --heuristic path/to/h.py
    python python_scripts/llm/MoH/eval_moh_atsp.py --task atsp_gls --seed-heuristic

`--every-subtask` is worth knowing about: a MoH run is multi-task and produces
one heuristic per size, so "the result of this run" is a family rather than a
single function. The default scores the largest subtask's heuristic — the one
Eq. (2) weights most, and the one the generalisation claim is about — and writes
it as `eval_test`; `--every-subtask` scores them all into
`eval_test_<subtask>.json` so you can see the size trade-off directly.

Which instances are scored, and which were searched on, both come from the
active data config (`configs/llm/MoH/cfg/data/<ATSP_DATA>.yaml`, default
`synthetic`). Under that default nothing overlaps: MoH evolves on generated
matrices only, so all nineteen TSPLIB instances are held out. Under
`ATSP_DATA=tsplib` two of them are in the search split, and `--exclude-train`
gives the honest number.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "solvers", "llm", "MoH")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.llm.MoH import (  # noqa: E402
    TASKS, load_heuristic_for_run, run_benchmark, seed_heuristic, subtasks_of_run,
    write_results,
)
from evaluation.metrics import markdown_table  # noqa: E402
from atsp.data import resolve_split  # noqa: E402
from atsp_utils import data_config_name, test_split, trained_on  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "llm", "MoH")


def evaluate(task: str, code: str, label: str, out_dir: str, name: str,
             max_n: int | None, names: list[str] | None,
             exclude_train: bool, subtask: str | None = None) -> dict:
    # The reported set is the `test` split of the active data config, so
    # changing configs/llm/MoH/cfg/data/<name>.yaml changes what is reported as
    # well as what is searched on — they cannot drift apart.
    #
    # A split is a single spec OR a list of them (cfg/data/smoke.yaml is a list),
    # so narrow every spec rather than assuming one.
    raw = test_split()
    specs = [dict(one) for one in
             (raw if isinstance(raw, (list, tuple)) else [raw])]
    for spec in specs:
        if max_n:
            spec["max_n"] = max_n
        if names:
            spec["names"] = names
    instances = resolve_split(specs if len(specs) > 1 else specs[0], ROOT,
                              log=lambda *a: None)

    trained = trained_on()
    if exclude_train and trained:
        instances = [i for i in instances if i.name not in trained]

    overlap = sorted(trained.intersection(i.name for i in instances))
    print(f"\n=== {task}: {label} on {len(instances)} TSPLIB instances ===")
    if overlap:
        print(f"    note: {', '.join(overlap)} are in the search split; "
              f"pass --exclude-train for the held-out number")

    records = run_benchmark(task, code, instances)
    written = write_results(out_dir, name, records, {
        "title": f"ATSP {task} — {label}", "framework": "MoH",
        "task": task, "heuristic": label, "split": "test", "subtask": subtask,
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


def _evaluate_run(run_dir: str, args) -> None:
    if args.every_subtask:
        for subtask in subtasks_of_run(run_dir):
            task, code = load_heuristic_for_run(run_dir, subtask)
            evaluate(task, code, f"best_heuristic_{subtask}.py",
                     args.out or run_dir, args.name or f"eval_test_{subtask}",
                     args.max_n, args.names, args.exclude_train, subtask)
        return
    task, code = load_heuristic_for_run(run_dir, args.subtask)
    label = f"best_heuristic_{args.subtask}.py" if args.subtask else "best_heuristic.py"
    default_name = f"eval_test_{args.subtask}" if args.subtask else "eval_test"
    evaluate(task, code, label, args.out or run_dir, args.name or default_name,
             args.max_n, args.names, args.exclude_train, args.subtask)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="a MoH run directory")
    parser.add_argument("--all", action="store_true",
                        help="every finished run under runs/llm/MoH/")
    parser.add_argument("--subtask", help="score this subtask's heuristic, "
                                          "e.g. atsp_gls-50")
    parser.add_argument("--every-subtask", action="store_true",
                        help="score every subtask's heuristic separately")
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--heuristic", help="path to a heuristic .py file")
    parser.add_argument("--seed-heuristic", action="store_true",
                        help="score the task's seed function (the baseline)")
    parser.add_argument("--exclude-train", action="store_true",
                        help="drop the instances the search was scored on")
    parser.add_argument("--max-n", type=int)
    parser.add_argument("--names", nargs="*")
    # A partial sweep (--names/--max-n) writes the same filenames as a full one
    # and would silently replace it. Send exploratory runs somewhere else.
    parser.add_argument("--out", help="output directory (default: the run dir, "
                                      "else runs/llm/MoH/eval/<task>)")
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
            _evaluate_run(os.path.dirname(path), args)
        return 0

    if args.run:
        run_dir = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
        _evaluate_run(run_dir, args)
        return 0

    if not args.task:
        parser.error("provide --run, --all, or --task")

    if args.seed_heuristic or not args.heuristic:
        # The label must start with "baseline" — that prefix is how
        # evaluation/llm/EoH/stats_analysis.py tells a hand-written reference
        # heuristic from a designed one when it builds the comparison table.
        code, label = seed_heuristic(args.task), "baseline: MoH seed function"
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
