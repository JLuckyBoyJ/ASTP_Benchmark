#!/usr/bin/env python
"""Build the paper tables from every evaluated run in the repository.

Repo-level driver: it assembles one section per solver family. Today that is
EoH (`evaluation/llm/EoH`); exact, heuristic and neural families slot in the
same way.

    python python_scripts/generate_paper_tables.py
    python python_scripts/generate_paper_tables.py --out paper/supplementary/results.md
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solvers.llm.EoH.atsp.config import repo_root  # noqa: E402
from solvers.llm.EoH.atsp.registry import TASK_SUMMARY  # noqa: E402

sys.path.insert(0, os.path.join(repo_root(), "python_scripts", "llm", "EoH"))
from generate_eoh_tables import build_report  # noqa: E402

ROOT = repo_root()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASK_SUMMARY),
                        help="restrict the EoH section to one task")
    parser.add_argument("--out", default=os.path.join("runs", "benchmark_tables.md"))
    args = parser.parse_args(argv)

    report, n_rows = build_report(os.path.join(ROOT, "runs", "llm", "EoH"), args.task)
    if not n_rows:
        print("Nothing to tabulate yet — run python_scripts/run_benchmarks.py first.")
        return 1

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(report)
    print(f"\nWritten to {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
