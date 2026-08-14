#!/usr/bin/env python
"""Materialise the instances an MCTS-AHD data config asks for.

`configs/llm/MCTS-AHD/cfg/data/<name>.yaml` declares train / val / test splits.
TSPLIB specs need nothing — the files are in `data/raw/atsp`. Synthetic specs
need generating once, together with the reference cost each instance is scored
against, which is the slow part. This script does that and caches the result in
`data/synthetic/`, where the EoH and ReEvo sides read from too, so a set is
computed once for the whole repository.

    python python_scripts/llm/MCTS-AHD/prepare_data.py                     # the default config
    python python_scripts/llm/MCTS-AHD/prepare_data.py --config synthetic
    python python_scripts/llm/MCTS-AHD/prepare_data.py --config mcts_ahd_native
    python python_scripts/llm/MCTS-AHD/prepare_data.py --all               # every config
    python python_scripts/llm/MCTS-AHD/prepare_data.py --config synthetic --force

`data/generate_atsp.py --all` remains the EoH-side generator and is unchanged;
this one only builds what the MCTS-AHD configs reference, so adding a data
config here never lengthens somebody else's setup.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOLVER = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")
for _p in (ROOT, SOLVER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atsp.data.synthetic import (  # noqa: E402
    dataset_filename, generate_dataset, save_dataset,
)
from atsp_utils import CONFIG_ROOT, load_config  # noqa: E402

SYNTHETIC_DIR = os.path.join(ROOT, "data", "synthetic")


def specs_of(config: dict) -> list[dict]:
    out = []
    for mood in ("train", "val", "test"):
        spec = config.get(mood)
        for one in (spec if isinstance(spec, (list, tuple)) else [spec]):
            if one and str(one.get("source", "synthetic")).lower() == "synthetic":
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


def main(argv=None) -> int:
    available = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(CONFIG_ROOT, "data", "*.yaml")))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=available, default="tsplib")
    parser.add_argument("--all", action="store_true",
                        help="prepare every data config")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if the cache file exists")
    args = parser.parse_args(argv)

    names = available if args.all else [args.config]
    total = 0
    for name in names:
        config = load_config("data", name)
        specs = specs_of(config)
        print(f"\n=== {name} ===")
        if not specs:
            print("  nothing to generate (TSPLIB only — files are in data/raw/atsp)")
            continue
        for spec in specs:
            build(spec, args.force)
            total += 1

    print(f"\nDone. {total} synthetic set(s) checked or built under "
          f"{os.path.relpath(SYNTHETIC_DIR, ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
