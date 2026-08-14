#!/usr/bin/env python
"""Verify ATSP datasets for HSEvo training and evaluation."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "solvers", "llm", "HSEvo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atsp_utils import load_instances


def main():
    print("[*] Verifying HSEvo training instances...")
    train_insts = load_instances("train", None)
    print(f"Loaded {len(train_insts)} training instances.")

    print("\n[*] Verifying HSEvo TSPLIB validation/test instances...")
    val_insts = load_instances("val", None)
    print(f"Loaded {len(val_insts)} TSPLIB validation instances.")

    print("\n[+] Dataset verification complete.")


if __name__ == "__main__":
    main()
