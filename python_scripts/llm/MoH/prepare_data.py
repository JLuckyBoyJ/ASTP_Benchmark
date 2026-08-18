#!/usr/bin/env python
"""Materialise the instances a MoH data config asks for.

`configs/llm/MoH/cfg/data/<name>.yaml` declares the search split and the test
split. TSPLIB specs need nothing — the files are in `data/raw/atsp`. Synthetic
specs need generating once, together with the reference cost each instance is
scored against, which is the slow part. This script does that and caches the
result in `data/synthetic/`, where every other solver family reads from too, so
a set is computed once for the whole repository.

    python python_scripts/llm/MoH/prepare_data.py                    # the default config
    python python_scripts/llm/MoH/prepare_data.py --config multisize
    python python_scripts/llm/MoH/prepare_data.py --all              # every config
    python python_scripts/llm/MoH/prepare_data.py --config synthetic --force

It also **checks the sizes**, which matters more here than for the other
frameworks: MoH's subtasks are instance sizes, so a data config and a problem
config disagree the moment one of them names a size the other lacks. The run
would fail at the first evaluation, after paying for the seeding prompts; this
catches it beforehand and costs nothing.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOLVER = os.path.join(ROOT, "solvers", "llm", "MoH")
for _p in (ROOT, SOLVER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from atsp.data.synthetic import (  # noqa: E402
    dataset_filename, generate_dataset, save_dataset,
)
from atsp_utils import CONFIG_ROOT, TASKS, load_config  # noqa: E402

SYNTHETIC_DIR = os.path.join(ROOT, "data", "synthetic")


def specs_of(config: dict) -> list[dict]:
    """Every synthetic spec in a config, de-duplicated across splits.

    `train` is a YAML anchor to `val` in the MoH configs — MoH has one in-search
    split — so without de-duplication every set would be reported twice.
    """
    out, seen = [], set()
    for mood in ("train", "val", "test"):
        spec = config.get(mood)
        for one in (spec if isinstance(spec, (list, tuple)) else [spec]):
            if not one or str(one.get("source", "synthetic")).lower() != "synthetic":
                continue
            key = (one.get("family", "uniform"), one.get("size", 50),
                   one.get("count", 8), one.get("seed", 2024))
            if key not in seen:
                seen.add(key)
                out.append(one)
    return out


def build(spec: dict, force: bool) -> str:
    family = str(spec.get("family", "uniform"))
    size = int(spec.get("size", 50))
    count = int(spec.get("count", 8))
    seed = int(spec.get("seed", 2024))
    effort = str(spec.get("effort", "medium"))

    out_dir = os.path.join(SYNTHETIC_DIR, family)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, dataset_filename(family, size, count, seed))

    if os.path.exists(path) and not force:
        print(f"  exists, skipping: {os.path.relpath(path, ROOT)}")
        return path

    started = time.time()

    def progress(done, total, instance):
        print(f"    [{done:>3}/{total}] {instance.name}  "
              f"reference={instance.ref_cost:,.0f}")

    print(f"  generating {count} x {family} n={size} (seed={seed}, effort={effort})")
    instances = generate_dataset(family, size, count, seed, effort=effort,
                                 progress=progress)
    save_dataset(path, instances)
    print(f"  -> {os.path.relpath(path, ROOT)}  ({time.time() - started:.1f}s)")
    return path


def check_sizes(config_name: str, config: dict) -> list[str]:
    """Warn where a problem config names a subtask size this split lacks."""
    sizes = {int(s.get("size")) for s in specs_of(config) if s.get("size")}
    if not sizes:
        return []          # TSPLIB-only split; sizes come from the instances
    problems = []
    for task in TASKS:
        path = os.path.join(ROOT, "configs", "llm", "MoH", "cfg", "problem",
                            f"{task}.yaml")
        if not os.path.isfile(path):
            continue
        wanted = yaml.safe_load(open(path, encoding="utf-8")).get("problem_size") or []
        missing = sorted({int(s) for s in wanted} - sizes)
        if missing:
            problems.append(
                f"  {task}: problem_size wants {missing}, which data/{config_name}.yaml "
                f"does not contain (it has {sorted(sizes)}). Either add a spec of that "
                f"size or pass problem.problem_size on the command line.")
    return problems


def main(argv=None) -> int:
    available = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(CONFIG_ROOT, "data", "*.yaml")))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=available, default="synthetic")
    parser.add_argument("--all", action="store_true", help="prepare every data config")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if the cache file exists")
    parser.add_argument("--check-only", action="store_true",
                        help="only report size mismatches; generate nothing")
    args = parser.parse_args(argv)

    names = available if args.all else [args.config]
    total, warnings = 0, []
    for name in names:
        config = load_config("data", name)
        specs = specs_of(config)
        print(f"\n=== {name} ===")
        problems = check_sizes(name, config)
        if problems:
            print("  size mismatch with the problem configs:")
            for line in problems:
                print(line)
            warnings.extend(problems)
        if not specs:
            print("  nothing to generate (TSPLIB only — files are in data/raw/atsp)")
            continue
        if args.check_only:
            for spec in specs:
                print(f"  would build {spec}")
            continue
        for spec in specs:
            build(spec, args.force)
            total += 1

    print(f"\nDone. {total} synthetic set(s) checked or built under "
          f"{os.path.relpath(SYNTHETIC_DIR, ROOT)}.")
    if warnings:
        print(f"{len(warnings)} size mismatch(es) above — a run using that "
              f"combination will stop at its first evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
