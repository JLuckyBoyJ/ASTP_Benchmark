"""Score a ReEvo-designed ATSP heuristic on a set of instances.

Mirrors `evaluation/llm/EoH/benchmark_runner.py` and writes the same files, so
`evaluation/llm/EoH/stats_analysis.py` can aggregate both frameworks into one
table. Instances come from ReEvo's own `atsp/` package. The difference is only in *how a heuristic is invoked*: ReEvo's tasks
hand the LLM a whole matrix (`heuristics`) or a step rule (`select_next_node`),
and are evaluated through ReEvo's own engines under `solvers/llm/ReEvo/problems`.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REEVO = os.path.join(ROOT, "solvers", "llm", "ReEvo")
for _p in (ROOT, REEVO, os.path.join(REEVO, "problems", "atsp_gls"),
           os.path.join(REEVO, "problems", "atsp_aco")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.metrics import markdown_table, summarise  # noqa: E402

CSV_FIELDS = ["instance", "n", "cost", "ref_cost", "ref_kind", "gap_percent",
              "seconds", "status", "error"]

#: task -> (entry point the LLM writes, evaluation budget)
TASKS = {
    "atsp_gls": {"entry": "heuristics",
                 "params": {"perturbation_moves": 30, "iter_limit": 1000,
                            "time_limit": 10.0}},
    "atsp_aco": {"entry": "heuristics",
                 "params": {"n_iterations": 50, "n_ants": 20, "time_limit": 30.0}},
    "atsp_constructive": {"entry": "select_next_node", "params": {}},
}


def load_callable(code: str, entry_point: str):
    """Compile a heuristic source string and return its entry point.

    ReEvo names functions `<name>_v1`, `_v2`, ... during evolution, so accept
    any of those and fall back to the bare name.
    """
    module = types.ModuleType("reevo_heuristic")
    module.__dict__["np"] = np
    exec(compile(code, "<heuristic>", "exec"), module.__dict__)
    for name in (f"{entry_point}_v3", f"{entry_point}_v2",
                 f"{entry_point}_v1", entry_point):
        func = getattr(module, name, None)
        if callable(func):
            return func
    raise ValueError(f"no {entry_point}[_vN] defined in the heuristic source")


def seed_heuristic(task: str) -> str:
    """ReEvo's own seed function — the baseline every task must beat."""
    path = os.path.join(REEVO, "prompts", task, "seed_func.txt")
    with open(path, encoding="utf-8") as fh:
        return "import numpy as np\n\n" + fh.read().replace("_v1(", "(")


def load_heuristic_for_run(run_dir: str) -> tuple[str, str]:
    """Return ``(task, source_code)`` for a finished ReEvo run directory."""
    best = os.path.join(run_dir, "best_heuristic.py")
    if not os.path.exists(best):
        raise SystemExit(f"{run_dir} has no best_heuristic.py — did the run finish?")
    meta_path = os.path.join(run_dir, "meta.json")
    task = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            task = json.load(fh).get("problem")
    if task not in TASKS:                      # fall back to the directory name
        task = os.path.basename(os.path.dirname(run_dir)).split("-")[0]
    if task not in TASKS:
        raise SystemExit(f"cannot tell which task {run_dir} belongs to")
    with open(best, encoding="utf-8") as fh:
        return task, fh.read()


# ── per-task solvers ─────────────────────────────────────────────────────────

def _solve_gls(func, instance, params) -> float:
    from gls import guided_local_search
    dist = instance.dist
    guide = func(dist.copy())
    _, cost = guided_local_search(dist, guide, **params)
    return cost


def _solve_aco(func, instance, params) -> float:
    from aco import ACO
    params = dict(params)
    n_iterations = params.pop("n_iterations")
    n_ants = params.pop("n_ants")
    time_limit = params.pop("time_limit", None)
    dist = instance.dist.copy()
    np.fill_diagonal(dist, 1.0)       # upstream guard against 1/0 heuristics
    aco = ACO(instance.dist, func(dist), n_ants=n_ants, seed=2024)
    deadline = None if time_limit is None else time.perf_counter() + time_limit
    return float(aco.run(n_iterations, deadline=deadline))


def _solve_constructive(func, instance, params) -> float:
    from copy import copy
    n, dist = instance.n, instance.dist
    solution = [0]
    unvisited = set(range(n))
    unvisited.remove(0)
    for _ in range(n - 1):
        nxt = int(func(current_node=solution[-1], destination_node=0,
                       unvisited_nodes=copy(unvisited),
                       distance_matrix=dist.copy()))
        if nxt not in unvisited:
            raise ValueError(f"select_next_node returned an invalid node: {nxt}")
        solution.append(nxt)
        unvisited.remove(nxt)
    tour = np.asarray(solution, dtype=np.int64)
    return float(dist[tour, np.roll(tour, -1)].sum())


SOLVERS = {"atsp_gls": _solve_gls, "atsp_aco": _solve_aco,
           "atsp_constructive": _solve_constructive}


# ── driver ───────────────────────────────────────────────────────────────────

def run_benchmark(task: str, heuristic_code: str, instances,
                  params: dict | None = None, log=print) -> list[dict]:
    if task not in TASKS:
        raise ValueError(f"unknown ReEvo task {task!r}; expected {sorted(TASKS)}")
    func = load_callable(heuristic_code, TASKS[task]["entry"])
    params = TASKS[task]["params"] if params is None else params
    solve = SOLVERS[task]

    records = []
    for instance in instances:
        started = time.time()
        record = {"instance": instance.name, "n": instance.n, "cost": None,
                  "ref_cost": instance.ref_cost, "ref_kind": instance.ref_kind,
                  "gap_percent": None, "seconds": 0.0, "status": "ok", "error": None}
        try:
            cost = float(solve(func, instance, params))
            record["cost"] = cost
            record["gap_percent"] = instance.gap(cost)
        except Exception as exc:          # a broken heuristic must not abort the sweep
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


def write_results(out_dir: str, name: str, records: list[dict], meta: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in CSV_FIELDS})

    payload = {**meta, "summary": summarise(records), "records": records}
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {meta.get('title', name)}\n\n")
        for key in ("framework", "task", "heuristic", "split"):
            if meta.get(key):
                fh.write(f"- **{key}**: {meta[key]}\n")
        summary = payload["summary"]
        fh.write(f"\n**mean gap {summary['mean_gap_percent']:.3f}%** over "
                 f"{summary['n_solved']}/{summary['n_instances']} instances "
                 f"({summary['n_optimal']} optimal)\n\n")
        fh.write(markdown_table(records) + "\n")

    return {"csv": csv_path, "json": os.path.join(out_dir, f"{name}.json"),
            "markdown": md_path, "summary": payload["summary"]}
