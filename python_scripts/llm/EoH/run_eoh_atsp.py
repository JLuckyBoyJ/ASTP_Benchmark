#!/usr/bin/env python
"""Evolve an ATSP heuristic with EoH.

Examples
--------
    # one task, default settings (gpt-4o-mini)
    python python_scripts/llm/EoH/run_eoh_atsp.py --task gls

    # explicit config + overrides + a run tag
    python python_scripts/llm/EoH/run_eoh_atsp.py \
        --config configs/llm/EoH/atsp_construct.yaml \
        --set eoh.n_pop=5 --set data.train.count=4 --tag pilot

    # check data, engine and timing without spending a single LLM call
    python python_scripts/llm/EoH/run_eoh_atsp.py --task rnr --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))

from solvers.llm.EoH.atsp.config import load_config, repo_root  # noqa: E402
from solvers.llm.EoH.atsp.registry import TASK_SUMMARY  # noqa: E402
from solvers.llm.EoH.atsp import runner  # noqa: E402

CONFIG_DIR = os.path.join("configs", "llm", "EoH")


def default_config_for(task: str) -> str:
    return os.path.join(repo_root(), CONFIG_DIR, f"atsp_{task}.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EoH (Evolution of Heuristics) on the ATSP benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_SUMMARY),
                        help="shortcut for --config configs/llm/EoH/atsp_<task>.yaml")
    parser.add_argument("--config", help="path to a YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override any config key, e.g. --set eoh.n_pop=5")
    parser.add_argument("--tag", help="suffix appended to the run directory name")
    parser.add_argument("--model", help="shortcut for --set llm.model=...")
    parser.add_argument("--debug", action="store_true",
                        help="DEBUG logging, including raw LLM output")
    parser.add_argument("--smoke", action="store_true",
                        help="evaluate the hand-crafted baseline once; no LLM calls")
    parser.add_argument("--list-tasks", action="store_true", help="list tasks and exit")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_tasks:
        print("ATSP tasks available for EoH:\n")
        for name, summary in sorted(TASK_SUMMARY.items()):
            print(f"  {name:<10} {summary}")
            print(f"  {'':<10} config: {CONFIG_DIR}/atsp_{name}.yaml")
        return 0

    if not args.config and not args.task:
        build_parser().error("provide --task or --config")
    config_path = args.config or default_config_for(args.task)
    if not os.path.exists(config_path):
        build_parser().error(f"config not found: {config_path}")

    overrides = list(args.overrides)
    if args.tag:
        overrides.append(f"run.tag={args.tag}")
    if args.model:
        overrides.append(f"llm.model={args.model}")
    if args.debug:
        overrides.append("eoh.debug=true")

    config = load_config(config_path, overrides)
    if args.task and config["task"]["name"] != args.task:
        config["task"]["name"] = args.task

    summary = runner.smoke(config) if args.smoke else runner.evolve(config)
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
