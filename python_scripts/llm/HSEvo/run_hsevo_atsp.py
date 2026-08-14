#!/usr/bin/env python
"""Wrapper CLI to launch HSEvo ATSP evolution from the repository root.

Examples:
    # Run default task (atsp_gls)
    python python_scripts/llm/HSEvo/run_hsevo_atsp.py

    # Run specific task with custom model and budget
    python python_scripts/llm/HSEvo/run_hsevo_atsp.py --task atsp_aco --model gpt-4o-mini --max-fe 50
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HSEVO_DIR = os.path.join(ROOT, "solvers", "llm", "HSEvo")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="atsp_gls", choices=["atsp_gls", "atsp_constructive", "atsp_aco"],
                        help="ATSP task to run (default: atsp_gls)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--max-fe", type=int, default=None,
                        help="Maximum function evaluations")
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Population size")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional Hydra overrides passed directly to main.py")

    args = parser.parse_args(argv)

    cmd = [sys.executable, "main.py", f"problem={args.task}", f"model={args.model}"]
    if args.max_fe is not None:
        cmd.append(f"max_fe={args.max_fe}")
    if args.pop_size is not None:
        cmd.append(f"pop_size={args.pop_size}")
    if args.seed is not None:
        cmd.append(f"seed={args.seed}")
    cmd.extend(args.extra_args)

    print(f"[*] Launching HSEvo in {HSEVO_DIR}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=HSEVO_DIR)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
