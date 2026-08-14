#!/usr/bin/env python
"""One line per MCTS-AHD run — the index into runs/llm/MCTS-AHD/.

Every run writes its own `run.log`, `summary.json` and `mcts_tree.json`, which
is what you want when inspecting *one* run and useless when you have thirty.
This walks them all and prints a table, so you can see at a glance which run
was best, which ones failed, and what each cost.

    python python_scripts/llm/MCTS-AHD/list_runs.py
    python python_scripts/llm/MCTS-AHD/list_runs.py --task atsp_gls
    python python_scripts/llm/MCTS-AHD/list_runs.py --sort objective --csv runs/index.csv
    python python_scripts/llm/MCTS-AHD/list_runs.py --unfinished     # what died

A run counts as finished when it has `summary.json`; an unfinished one is shown
with whatever `progress.jsonl` got to before it stopped, which is usually enough
to tell a crash from a Ctrl-C.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "llm", "MCTS-AHD")


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _last_progress(run_dir: str) -> dict:
    path = os.path.join(run_dir, "progress.jsonl")
    last = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = json.loads(line)
    except Exception:
        pass
    return last


def collect(task: str | None = None) -> list[dict]:
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(RUNS, "*", "*", "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        meta = _read_json(meta_path)
        if task and meta.get("problem") != task:
            continue
        summary = _read_json(os.path.join(run_dir, "summary.json"))
        progress = _last_progress(run_dir)
        evaluation = _read_json(os.path.join(run_dir, "eval_test.json"))

        finished = bool(summary)
        rows.append({
            "run": os.path.relpath(run_dir, RUNS),
            "task": meta.get("problem", "?"),
            "model": meta.get("model", "?"),
            "data": (meta.get("config", {}) or {}).get("data", "?"),
            "max_fe": meta.get("max_fe"),
            "started": meta.get("started_at", ""),
            "status": "ok" if finished else "unfinished",
            "train_objective": (summary.get("best_objective")
                                if finished else progress.get("best_so_far")),
            "test_gap": (evaluation.get("summary", {}) or {}).get("mean_gap_percent"),
            "evals": (summary.get("function_evals")
                      if finished else progress.get("eval")),
            "nodes": summary.get("n_tree_nodes"),
            "depth": summary.get("max_tree_depth"),
            "minutes": summary.get("minutes"),
            "llm_calls": summary.get("llm_calls"),
            "run_dir": run_dir,
        })
    return rows


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return "-"
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def table(rows: list[dict]) -> str:
    header = ["run", "task", "model", "data", "status", "train obj",
              "test gap %", "evals", "nodes", "depth", "min", "calls"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["run"], r["task"], r["model"], str(r["data"]), r["status"],
            _fmt(r["train_objective"], ".3f"), _fmt(r["test_gap"], ".3f"),
            _fmt(r["evals"]), _fmt(r["nodes"]), _fmt(r["depth"]),
            _fmt(r["minutes"], ".1f"), _fmt(r["llm_calls"]),
        ]) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task")
    parser.add_argument("--sort", choices=["started", "objective", "test", "task"],
                        default="started")
    parser.add_argument("--unfinished", action="store_true",
                        help="show only runs with no summary.json")
    parser.add_argument("--csv", help="also write the table to this path")
    args = parser.parse_args(argv)

    rows = collect(args.task)
    if args.unfinished:
        rows = [r for r in rows if r["status"] != "ok"]
    if not rows:
        print(f"No runs under {os.path.relpath(RUNS, ROOT)}.")
        return 1

    keys = {
        "started": lambda r: r["started"],
        "task": lambda r: (r["task"], r["started"]),
        "objective": lambda r: (r["train_objective"] is None, r["train_objective"]),
        "test": lambda r: (r["test_gap"] is None, r["test_gap"]),
    }
    rows.sort(key=keys[args.sort])

    print(table(rows))
    finished = sum(r["status"] == "ok" for r in rows)
    print(f"\n{len(rows)} run(s), {finished} finished. "
          f"Logs: runs/llm/MCTS-AHD/<task>/<timestamp>/run.log")
    if any(r["test_gap"] is None for r in rows):
        print("Runs with no test gap have not been benchmarked yet:\n"
              "  bash scripts/llm/MCTS-AHD/benchmark.sh")

    if args.csv:
        out = args.csv if os.path.isabs(args.csv) else os.path.join(ROOT, args.csv)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written to {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
