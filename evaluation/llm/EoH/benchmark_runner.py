"""Run one heuristic over a set of ATSP instances and score it.

The same code path serves EoH-evolved heuristics and the hand-crafted
baselines, so their numbers are directly comparable. Results are written as
CSV + JSON next to the run they came from.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))

from solvers.llm.EoH.atsp.baselines import baseline_code, baseline_name, load_callable
from solvers.llm.EoH.atsp.config import repo_root
from solvers.llm.EoH.atsp.data import resolve_split
from solvers.llm.EoH.atsp.registry import build_problem, get_problem_class

from evaluation.metrics import markdown_table, summarise

CSV_FIELDS = ["instance", "n", "cost", "ref_cost", "ref_kind", "gap_percent",
              "seconds", "status", "error"]


def read_heuristic(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def resolve_heuristic(task: str, heuristic_path: str | None,
                      use_baseline: bool) -> tuple[str, str]:
    """Return ``(source_code, label)`` for the heuristic under evaluation."""
    if use_baseline or not heuristic_path:
        return baseline_code(task), f"baseline:{baseline_name(task)}"
    return read_heuristic(heuristic_path), os.path.basename(heuristic_path)


def run_benchmark(task: str, heuristic_code: str, instances, params: dict | None = None,
                  log=print) -> list[dict]:
    """Evaluate ``heuristic_code`` on every instance; never raises on a bad heuristic."""
    problem_cls = get_problem_class(task)
    entry_point = problem_cls.entry_point
    func = load_callable(heuristic_code, entry_point)

    records: list[dict] = []
    for instance in instances:
        problem = build_problem(task, [instance], params, n_processes=1)
        started = time.time()
        record = {
            "instance": instance.name, "n": instance.n, "cost": None,
            "ref_cost": instance.ref_cost, "ref_kind": instance.ref_kind,
            "gap_percent": None, "seconds": 0.0, "status": "ok", "error": None,
        }
        try:
            cost = float(problem.solve_instance(func, instance))
            record["cost"] = cost
            record["gap_percent"] = instance.gap(cost)
        except Exception as exc:  # a broken heuristic must not abort the sweep
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.time() - started, 2)
        records.append(record)

        gap = record["gap_percent"]
        gap_s = "   n/a " if gap is None else f"{gap:7.3f}"
        log(f"  {instance.name:<12} n={instance.n:<5} "
            f"cost={'failed' if record['cost'] is None else format(record['cost'], ',.0f'):>12}  "
            f"gap={gap_s}%  ({record['seconds']:.1f}s)"
            + (f"  [{record['error']}]" if record["error"] else ""))
    return records


def load_instances(config: dict, split: str = "test", log=print):
    return resolve_split(config["data"][split], repo_root(), log=log)


def write_results(out_dir: str, name: str, records: list[dict], meta: dict) -> dict:
    # Which framework produced the heuristic. ReEvo's runner stamps its own;
    # anything written here came from EoH. `stats_analysis` groups on this, so
    # a merged table keeps the two apart instead of calling everything "EoH".
    meta = {"framework": "EoH", **meta}
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in CSV_FIELDS})

    payload = {**meta, "summary": summarise(records), "records": records}
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {meta.get('title', name)}\n\n")
        for key in ("task", "heuristic", "split", "model", "run_dir"):
            if meta.get(key):
                fh.write(f"- **{key}**: {meta[key]}\n")
        summary = payload["summary"]
        fh.write(f"\n**mean gap {summary['mean_gap_percent']:.3f}%** over "
                 f"{summary['n_solved']}/{summary['n_instances']} instances "
                 f"({summary['n_optimal']} optimal)\n\n")
        fh.write(markdown_table(records) + "\n")

    return {"csv": csv_path, "json": json_path, "markdown": md_path,
            "summary": payload["summary"]}
