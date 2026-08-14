#!/usr/bin/env python
"""Launch one MCTS-AHD run from the repository root.

MCTS-AHD's own entry point is `solvers/llm/MCTS-AHD/main.py`, and it must be
launched from that directory because Hydra resolves `problems/` and `prompts/`
relative to the working directory. That is fine for one run and annoying for
everything else, so this wrapper does the `cd` for you, checks the obvious
preconditions before spending money, and prints where the run landed.

    python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls
    python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_kgls --max-fe 1000
    python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_aco --data synthetic
    python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --smoke
    python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --set exploration_constant=0.05

`--smoke` is a free dry run: the offline stub model, a budget of six
evaluations. It exercises the tree, the actions, the evaluation subprocess and
the whole run directory without touching a paid API, which is the check worth
running before a long job.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOLVER = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")
RUNS = os.path.join(ROOT, "runs", "llm", "MCTS-AHD")

TASKS = ("atsp_constructive", "atsp_gls", "atsp_kgls", "atsp_aco")
DATA_CONFIGS = ("tsplib", "synthetic", "mcts_ahd_native")


def _newest_run(task: str) -> str | None:
    import glob
    found = glob.glob(os.path.join(RUNS, f"{task}-*", "*", "meta.json"))
    if not found:
        return None
    return os.path.dirname(max(found, key=os.path.getmtime))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model", default=None, help="e.g. gpt-4o-mini, gpt-4o")
    parser.add_argument("--max-fe", type=int, default=None,
                        help="evaluation budget T (config default: 100)")
    parser.add_argument("--data", choices=DATA_CONFIGS, default=None,
                        help="which cfg/data/<name>.yaml split to use")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="free dry run: stub model, six evaluations")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="any other Hydra override (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command instead of running it")
    args = parser.parse_args(argv)

    overrides = [f"problem={args.task}"]
    if args.smoke:
        overrides += ["llm_client=stub", "max_fe=6", "init_pop_size=2"]
    else:
        if not os.environ.get("OPENAI_API_KEY") and not _dotenv_has_key():
            print("OPENAI_API_KEY is not set. Put it in envs/.env "
                  "(cp envs/.env.example envs/.env), or pass --smoke for a free "
                  "dry run.", file=sys.stderr)
            return 2
        if args.model:
            overrides.append(f"llm_client.model={args.model}")
    if args.max_fe is not None:
        overrides.append(f"max_fe={args.max_fe}")
    if args.data is not None:
        overrides.append(f"data={args.data}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    overrides += list(args.overrides)

    cmd = [sys.executable, "main.py", *overrides]
    print("$ cd solvers/llm/MCTS-AHD && " + " ".join(cmd[1:]))
    if args.dry_run:
        return 0

    code = subprocess.call(cmd, cwd=SOLVER)

    run_dir = _newest_run(args.task)
    if run_dir:
        rel = os.path.relpath(run_dir, ROOT)
        print(f"\nRun directory: {rel}")
        print(f"  log      : {rel}/run.log")
        print(f"  tree     : {rel}/mcts_tree.json")
        print(f"  progress : {rel}/progress.jsonl")
        print(f"  winner   : {rel}/best_heuristic.py")
        print(f"\nScore it on the held-out benchmark with:\n"
              f"  python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py "
              f"--run {rel}")
    return code


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


if __name__ == "__main__":
    raise SystemExit(main())
