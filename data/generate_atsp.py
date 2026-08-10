#!/usr/bin/env python
"""Generate the synthetic ATSP training sets used to evolve heuristics.

EoH evolves on randomly generated instances and is benchmarked on held-out
TSPLIB instances, so nothing from ``data/raw/atsp`` is ever seen during the
search. This script materialises the random side of that split.

Each instance is stored with a *reference cost* (best tour found by a
deterministic multi-start Or-opt/swap local search) so the fitness can be an
optimality gap, as in the EoH paper.

Examples
--------
    # everything the shipped configs need
    python data/generate_atsp.py --all

    # a custom set
    python data/generate_atsp.py --family asymmetric_clustered --size 100 \
        --count 64 --seed 2024

Output: ``data/synthetic/<family>/atsp_<family>_n<size>_c<count>_s<seed>.npz``
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solvers.llm.EoH.atsp.data.synthetic import (  # noqa: E402
    FAMILIES,
    dataset_filename,
    generate_dataset,
    save_dataset,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYNTHETIC_DIR = os.path.join(REPO_ROOT, "data", "synthetic")

#: (family, size, count, seed, effort) — the union of every `data.train` entry
#: in configs/llm/EoH/*.yaml. The four tasks deliberately share these files, so
#: the reference-cost computation happens once rather than once per task.
DEFAULT_SETS = [
    ("asymmetric_clustered", 50, 8, 2024, "medium"),    # all four tasks
    ("uniform", 50, 8, 2024, "medium"),                 # construct
    ("uniform", 50, 4, 2024, "medium"),                 # rnr
    ("uniform", 50, 3, 2024, "medium"),                 # aco
    ("asymmetric_clustered", 200, 4, 2024, "low"),      # large + correlated
    ("uniform", 200, 4, 2024, "low"),                   # large + uncorrelated
]


def build(family: str, size: int, count: int, seed: int, force: bool = False,
          effort: str = "medium") -> str:
    out_dir = os.path.join(SYNTHETIC_DIR, family)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, dataset_filename(family, size, count, seed))

    if os.path.exists(path) and not force:
        print(f"  exists, skipping: {os.path.relpath(path, REPO_ROOT)}")
        return path

    started = time.time()

    def progress(done, total, instance):
        print(f"    [{done:>3}/{total}] {instance.name}  "
              f"reference={instance.ref_cost:,.0f}")

    print(f"  generating {count} x {family} n={size} (seed={seed}, effort={effort})")
    instances = generate_dataset(family, size, count, seed, effort=effort,
                                 progress=progress)
    save_dataset(path, instances)
    print(f"  -> {os.path.relpath(path, REPO_ROOT)}  ({time.time() - started:.1f}s)")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="generate every dataset referenced by configs/llm/EoH/")
    parser.add_argument("--family", choices=FAMILIES, default="uniform")
    parser.add_argument("--size", type=int, default=50, help="cities per instance")
    parser.add_argument("--count", type=int, default=8, help="number of instances")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="medium",
                        help="reference-solver budget (bigger = tighter gaps, slower)")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)

    print("Synthetic ATSP generation")
    if args.all:
        for family, size, count, seed, effort in DEFAULT_SETS:
            build(family, size, count, seed, args.force, effort)
    else:
        build(args.family, args.size, args.count, args.seed, args.force, args.effort)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
