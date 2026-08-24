#!/usr/bin/env python
"""Launch one MoH run from the repository root.

MoH's own entry point is `solvers/llm/MoH/main.py`, and it must be launched from
that directory because Hydra resolves `problems/` and `prompts/` relative to the
working directory. That is fine for one run and annoying for everything else, so
this wrapper does the `cd` for you, checks the obvious preconditions before
spending money, and prints where the run landed.

    python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls
    python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_kgls --iterations 20
    python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls --sizes 50 100 200
    python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls --data multisize
    python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls --smoke
    python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls --set pop_size=5

`--smoke` is a free dry run: the offline stub model for both loops, two tiny
subtasks (`data=smoke`), a 20-iteration engine budget (`evaluation_budget=smoke`)
and one outer iteration. It exercises the seeding, both loops, the evaluation
subprocess and the whole run directory in about a minute, without touching a
paid API — the check worth running before a long job. Nothing it produces is a
measurement.

Inference — apply an optimizer trained at one size to another (Section 3.2):

    python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls \\
        --inference runs/llm/MoH/atsp_gls-gls/<date>_<time>/best_meta_optimizer.py \\
        --sizes 400 --data multisize
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOLVER = os.path.join(ROOT, "solvers", "llm", "MoH")
RUNS = os.path.join(ROOT, "runs", "llm", "MoH")

TASKS = ("atsp_constructive", "atsp_gls", "atsp_kgls")
DATA_CONFIGS = ("synthetic", "tsplib", "multisize", "smoke")


def _newest_run(task: str) -> str | None:
    found = glob.glob(os.path.join(RUNS, f"{task}-*", "*", "meta.json"))
    if not found:
        return None
    return os.path.dirname(max(found, key=os.path.getmtime))


def _dotenv_has_key() -> bool:
    for name in (os.environ.get("ENV_FILE"), os.path.join(ROOT, "envs", ".env"),
                 os.path.join(ROOT, ".env")):
        if name and os.path.isfile(name):
            try:
                with open(name, encoding="utf-8") as fh:
                    for line in fh:
                        key = line.strip().removeprefix("export ").split("=", 1)[0]
                        if key.strip() == "OPENAI_API_KEY":
                            return True
            except OSError:
                continue
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model", default=None,
                        help="model for both loops, e.g. gpt-4o-mini, gpt-4o")
    parser.add_argument("--meta-model", default=None,
                        help="a different model for the outer loop only")
    parser.add_argument("--iterations", type=int, default=None,
                        help="outer-loop iterations T (config default: 15)")
    parser.add_argument("--pop-size", type=int, default=None,
                        help="population size (config default: 15)")
    parser.add_argument("--max-eval-calls", type=int, default=None,
                        help="heuristic evaluations per subtask (config default: 300)")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="the subtask sizes, e.g. --sizes 50 200. Every size "
                             "must exist in the chosen data config.")
    parser.add_argument("--data", choices=DATA_CONFIGS, default=None,
                        help="which cfg/data/<name>.yaml split to use")
    parser.add_argument("--inference", metavar="OPTIMIZER",
                        help="skip the outer loop and apply this trained "
                             "best_meta_optimizer.py (the paper's inference stage)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="free dry run: stub model, one iteration")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="any other Hydra override (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command instead of running it")
    args = parser.parse_args(argv)

    overrides = [f"problem={args.task}"]
    if args.smoke:
        # data=smoke and evaluation_budget=smoke are what make this finish in a
        # minute: two 20/40-city instances and a 20-iteration engine budget,
        # instead of the hour a real budget over real instances takes. It still
        # exercises two subtasks, so the size filtering and the size-weighted
        # utility are covered rather than skipped.
        overrides += ["llm_client@heu=stub", "llm_client@meta=stub",
                      "data=smoke", "evaluation_budget=smoke",
                      "problem.problem_size=[20,40]",
                      "n_iterations=1", "pop_size=3", "max_eval_calls=8",
                      "seed_rounds=1"]
    else:
        if not os.environ.get("OPENAI_API_KEY") and not _dotenv_has_key():
            print("OPENAI_API_KEY is not set. Put it in envs/.env "
                  "(cp envs/.env.example envs/.env), or pass --smoke for a free "
                  "dry run.", file=sys.stderr)
            return 2
        if args.model:
            overrides += [f"heu.model={args.model}", f"meta.model={args.model}"]
        if args.meta_model:
            overrides.append(f"meta.model={args.meta_model}")

    if args.iterations is not None:
        overrides.append(f"n_iterations={args.iterations}")
    if args.pop_size is not None:
        overrides.append(f"pop_size={args.pop_size}")
    if args.max_eval_calls is not None:
        overrides.append(f"max_eval_calls={args.max_eval_calls}")
    if args.sizes:
        overrides.append("problem.problem_size=[" + ",".join(map(str, args.sizes)) + "]")
    if args.data is not None:
        overrides.append(f"data={args.data}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.inference:
        path = args.inference if os.path.isabs(args.inference) \
            else os.path.join(ROOT, args.inference)
        if not os.path.isfile(path):
            print(f"no such optimizer: {args.inference}", file=sys.stderr)
            return 2
        overrides += ["mode=inference", f"meta_optimizer={path}"]
    overrides += list(args.overrides)

    cmd = [sys.executable, "main.py", *overrides]
    print("$ cd solvers/llm/MoH && " + " ".join(cmd[1:]))
    if args.dry_run:
        return 0

    code = subprocess.call(cmd, cwd=SOLVER)

    run_dir = _newest_run(args.task)
    if run_dir:
        rel = os.path.relpath(run_dir, ROOT)
        print(f"\nRun directory: {rel}")
        print(f"  log        : {rel}/run.log")
        print(f"  progress   : {rel}/progress.jsonl")
        print(f"  optimizer  : {rel}/best_meta_optimizer.py")
        print(f"  heuristics : {rel}/best_heuristic*.py")
        print(f"  prompts    : {rel}/llm_calls.jsonl")
        print(f"\nScore it on the held-out benchmark with:\n"
              f"  python python_scripts/llm/MoH/eval_moh_atsp.py --run {rel}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
