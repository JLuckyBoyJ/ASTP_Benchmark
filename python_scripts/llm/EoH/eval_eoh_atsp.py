#!/usr/bin/env python
"""Benchmark a heuristic on the held-out TSPLIB ATSP instances.

Examples
--------
    # evaluate the winner of a finished EoH run
    python python_scripts/llm/EoH/eval_eoh_atsp.py --run runs/llm/EoH/gls/20260810-142500

    # the hand-crafted baseline for the same task, for comparison
    python python_scripts/llm/EoH/eval_eoh_atsp.py --task gls --baseline

    # an arbitrary heuristic file, restricted to the smaller instances
    python python_scripts/llm/EoH/eval_eoh_atsp.py --task construct \
        --heuristic my_heuristic.py --max-n 100
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))

from solvers.llm.EoH.atsp.config import load_config, repo_root  # noqa: E402
from solvers.llm.EoH.atsp.logging_utils import (  # noqa: E402
    get_logger,
    read_json,
    setup_run_logging,
)
from solvers.llm.EoH.atsp.registry import TASK_SUMMARY  # noqa: E402

from evaluation.llm.EoH.benchmark_runner import (  # noqa: E402
    load_instances,
    resolve_heuristic,
    run_benchmark,
    write_results,
)
from evaluation.metrics import markdown_table  # noqa: E402

ROOT = repo_root()
logger = get_logger()


def _config_for(args) -> tuple[dict, str, str | None]:
    """Return ``(config, task, run_dir)`` for the requested evaluation."""
    if args.run:
        run_dir = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
        config_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise SystemExit(f"no config.yaml in {run_dir}")
        config = load_config(config_path, args.overrides)
        return config, config["task"]["name"], run_dir

    task = args.task
    if not task:
        raise SystemExit("provide --run or --task")
    config_path = args.config or os.path.join(ROOT, "configs", "llm", "EoH",
                                              f"atsp_{task}.yaml")
    return load_config(config_path, args.overrides), task, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="an EoH run directory (uses its best_heuristic.py)")
    parser.add_argument("--task", choices=sorted(TASK_SUMMARY))
    parser.add_argument("--config", help="config to take data/eval settings from")
    parser.add_argument("--heuristic", help="path to a heuristic .py file")
    parser.add_argument("--baseline", action="store_true",
                        help="evaluate the hand-crafted baseline instead")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--max-n", type=int, help="skip instances larger than this")
    parser.add_argument("--names", nargs="*", help="only these instances")
    parser.add_argument("--out", help="output directory (default: the run dir, else runs/llm/EoH/eval)")
    parser.add_argument("--name", help="basename of the result files")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    config, task, run_dir = _config_for(args)
    # A split may be a single spec or a list of specs; filters apply to each.
    split = config["data"][args.split]
    for spec in (split if isinstance(split, list) else [split]):
        if args.max_n is not None:
            spec["max_n"] = args.max_n
        if args.names:
            spec["names"] = args.names

    heuristic_path = args.heuristic
    train_objective = None
    if run_dir and not args.baseline and not heuristic_path:
        heuristic_path = os.path.join(run_dir, "best_heuristic.py")
        if not os.path.exists(heuristic_path):
            raise SystemExit(f"{run_dir} has no best_heuristic.py — did the run finish?")
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            train_objective = read_json(summary_path).get("best_objective")

    out_dir = args.out or run_dir or os.path.join(ROOT, "runs", "llm", "EoH", "eval", task)
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    setup_run_logging(out_dir, debug=False)

    code, label = resolve_heuristic(task, heuristic_path, args.baseline)
    params = config.get("eval", {}).get("params") or config["task"].get("params") or {}

    logger.info("=" * 62)
    logger.info("  ATSP benchmark — task '%s'", task)
    logger.info("  heuristic : %s", label)
    logger.info("  split     : %s", args.split)
    logger.info("  settings  : %s", params)
    logger.info("=" * 62)

    instances = load_instances(config, args.split, log=logger.info)
    logger.info("[data] %d instances", len(instances))

    records = run_benchmark(task, code, instances, params, log=logger.info)

    name = args.name or (f"eval_{args.split}" if not args.baseline
                         else f"eval_{args.split}_baseline")
    meta = {
        "title": f"ATSP {task} — {label} on {args.split}",
        "task": task, "heuristic": label, "split": args.split,
        "model": config["llm"].get("model"), "run_dir": run_dir,
        "params": params, "train_objective": train_objective,
    }
    written = write_results(out_dir, name, records, meta)

    logger.info("-" * 62)
    logger.info("\n%s", markdown_table(records))
    summary = written["summary"]
    logger.info("-" * 62)
    logger.info("  mean gap    : %.3f%%   (median %.3f%%)",
                summary["mean_gap_percent"], summary["median_gap_percent"])
    logger.info("  solved      : %d/%d instances, %d at the optimum",
                summary["n_solved"], summary["n_instances"], summary["n_optimal"])
    logger.info("  total time  : %.1fs", summary["total_seconds"])
    logger.info("  written     : %s", os.path.relpath(written["csv"], ROOT))
    logger.info("-" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
