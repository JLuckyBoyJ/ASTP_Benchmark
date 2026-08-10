"""Aggregate many EoH runs into the tables that go into the paper.

A "run" is one directory under ``runs/llm/EoH/<task>/<timestamp>/``. After
``eval_eoh_atsp.py`` has been executed for a run it contains
``eval_test.json``; this module collects those files across tasks, seeds and
methods and reduces them to mean +/- std tables.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))

from evaluation.metrics import comparison_table  # noqa: E402


def find_eval_files(runs_root: str, filename: str = "eval_test.json") -> list[str]:
    """Every evaluation file under ``runs_root``.

    Covers both ``<task>/<run>/`` (evaluations attached to a run) and
    ``eval/<task>/`` (standalone baseline evaluations).
    """
    patterns = [os.path.join(runs_root, "*", "*", filename),
                os.path.join(runs_root, "*", filename)]
    found = {path for pattern in patterns for path in glob.glob(pattern)}
    return sorted(found)


def load_eval(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def collect(runs_root: str, task: str | None = None,
            filename: str = "eval_test.json") -> list[dict]:
    """One row per evaluated run, optionally restricted to a single task.

    The task is read from the file's payload rather than from its path, so
    standalone baseline evaluations under ``eval/<task>/`` are filtered
    correctly too.
    """
    rows = []
    for path in find_eval_files(runs_root, filename):
        payload = load_eval(path)
        if task is not None and payload.get("task") != task:
            continue
        summary = payload.get("summary", {})
        rows.append({
            "task": payload.get("task"),
            "heuristic": payload.get("heuristic"),
            "model": payload.get("model"),
            "run_dir": payload.get("run_dir") or os.path.dirname(path),
            "train_objective": payload.get("train_objective"),
            "mean_gap_percent": summary.get("mean_gap_percent"),
            "median_gap_percent": summary.get("median_gap_percent"),
            "n_solved": summary.get("n_solved"),
            "n_instances": summary.get("n_instances"),
            "n_optimal": summary.get("n_optimal"),
            "total_seconds": summary.get("total_seconds"),
            "path": path,
        })
    return rows


def aggregate_by_task(rows: list[dict]) -> dict[str, dict]:
    """Mean / std / best of the test gap across repeated runs of each task."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        label = "baseline" if str(row.get("heuristic", "")).startswith("baseline") else "EoH"
        buckets[(row["task"], label)].append(row)

    out: dict[str, dict] = {}
    for (task, label), group in buckets.items():
        gaps = [r["mean_gap_percent"] for r in group
                if isinstance(r["mean_gap_percent"], (int, float))]
        out.setdefault(task, {})[label] = {
            "n_runs": len(group),
            "mean_gap_percent": mean(gaps) if gaps else float("nan"),
            "std_gap_percent": pstdev(gaps) if len(gaps) > 1 else 0.0,
            "best_gap_percent": min(gaps) if gaps else float("nan"),
            "runs": group,
        }
    return out


def summary_table(aggregated: dict[str, dict]) -> str:
    lines = ["| task | method | runs | mean gap % | std | best run gap % |",
             "|---|---|---:|---:|---:|---:|"]
    for task in sorted(aggregated):
        for label in sorted(aggregated[task], reverse=True):  # EoH before baseline
            entry = aggregated[task][label]
            lines.append(
                f"| {task} | {label} | {entry['n_runs']} | "
                f"{entry['mean_gap_percent']:.3f} | {entry['std_gap_percent']:.3f} | "
                f"{entry['best_gap_percent']:.3f} |")
    return "\n".join(lines)


def per_instance_comparison(rows: list[dict], task: str) -> str:
    """Instance-level EoH-vs-baseline table for a single task."""
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        if row["task"] != task:
            continue
        payload = load_eval(row["path"])
        label = ("baseline" if str(row.get("heuristic", "")).startswith("baseline")
                 else os.path.basename(str(row["run_dir"])))
        by_method[label] = payload.get("records", [])
    if not by_method:
        return f"_no evaluated runs for task {task}_"
    return comparison_table(by_method)
