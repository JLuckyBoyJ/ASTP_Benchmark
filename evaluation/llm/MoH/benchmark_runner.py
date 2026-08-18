"""Score a MoH-designed ATSP heuristic on a set of instances.

Mirrors ``evaluation/llm/MCTS-AHD/benchmark_runner.py`` and writes the same
files, so ``evaluation/llm/EoH/stats_analysis.py`` aggregates every framework
into one table. Instances come from MoH's own ``atsp/`` package, which is a copy
of the others', so the matrices and the reference costs are identical.

Budgets are **not** duplicated here: they are read from
``configs/llm/MoH/cfg/evaluation/<task>.yaml``, the same file each task's own
``eval.py`` reads, so a number produced by this runner and a number produced by
a run's post-hoc validation mean the same thing by construction rather than by
somebody remembering to keep two copies in step.

One difference from the MCTS-AHD runner, and it is the interesting one: MoH's
``atsp_gls`` heuristic is a **callable invoked every iteration**, not a static
matrix computed once. The engine in ``solvers/llm/MoH/problems/atsp_gls/gls.py``
takes the function; the MCTS-AHD engine takes the matrix. A heuristic from one
framework therefore cannot be scored by the other's runner, which is exactly
what the design-space difference means in practice. Everything downstream of
that — the instances, the local search, the objective — is shared.
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
MOH = os.path.join(ROOT, "solvers", "llm", "MoH")
for _p in (ROOT, MOH, os.path.join(MOH, "problems", "atsp_gls"),
           os.path.join(MOH, "problems", "atsp_kgls")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.metrics import markdown_table, summarise  # noqa: E402
from atsp_utils import budget as task_budget  # noqa: E402

CSV_FIELDS = ["instance", "n", "cost", "ref_cost", "ref_kind", "gap_percent",
              "seconds", "status", "error"]

#: task -> the function name the LLM writes. Budgets are not here: they come
#: from configs/llm/MoH/cfg/evaluation/<task>.yaml.
TASKS = {
    "atsp_constructive": {"entry": "select_next_node"},
    "atsp_gls": {"entry": "update_edge_distance"},
    "atsp_kgls": {"entry": "arc_badness"},
}

#: One line per task, for the report headers.
TASK_SUMMARY = {
    "atsp_constructive": "greedy construction — the LLM designs the next-city rule",
    "atsp_gls": "guided local search — the LLM designs a per-iteration arc-cost update",
    "atsp_kgls": "knowledge-guided local search — the LLM designs a dynamic arc-badness rule",
}


def default_params(task: str, mood: str = "test") -> dict:
    """The engine budget for ``task``, read from the shared YAML config."""
    return task_budget(task, mood)


def load_callable(code: str, entry_point: str):
    """Compile a heuristic source string and return its entry point.

    MoH keeps the function name fixed (the task prompt says "Do not change the
    function name"), but a model occasionally emits ``<name>_v2`` anyway, so the
    suffixed variants are accepted the way the other runners accept them.
    """
    module = types.ModuleType("moh_heuristic")
    module.__dict__["np"] = np
    exec(compile(code, "<heuristic>", "exec"), module.__dict__)
    for name in (f"{entry_point}_v3", f"{entry_point}_v2",
                 f"{entry_point}_v1", entry_point):
        func = getattr(module, name, None)
        if callable(func):
            return func
    raise ValueError(f"no {entry_point}[_vN] defined in the heuristic source")


def seed_heuristic(task: str) -> str:
    """The seed function of a task — the baseline it has to beat."""
    path = os.path.join(MOH, "prompts", task, "seed_func.txt")
    with open(path, encoding="utf-8") as fh:
        return "import numpy as np\n\n" + fh.read()


def load_heuristic_for_run(run_dir: str, subtask: str | None = None) -> tuple[str, str]:
    """Return ``(task, source_code)`` for a finished MoH run directory.

    Defaults to ``best_heuristic.py`` — the winner on the largest subtask, which
    is where every framework's benchmark scripts look. A MoH run also writes one
    ``best_heuristic_<task>-<size>.py`` per subtask, since a multi-task run
    produces a heuristic per size rather than a single winner; pass ``subtask``
    to score one of those instead.
    """
    name = f"best_heuristic_{subtask}.py" if subtask else "best_heuristic.py"
    best = os.path.join(run_dir, name)
    if not os.path.exists(best):
        available = sorted(f for f in os.listdir(run_dir)
                           if f.startswith("best_heuristic")) if os.path.isdir(run_dir) else []
        raise SystemExit(f"{run_dir} has no {name} — did the run finish? "
                         f"available: {available}")
    task = None
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            task = json.load(fh).get("problem")
    if task not in TASKS:                      # fall back to the directory name
        task = os.path.basename(os.path.dirname(run_dir)).split("-")[0]
    if task not in TASKS:
        raise SystemExit(f"cannot tell which task {run_dir} belongs to")
    with open(best, encoding="utf-8") as fh:
        return task, fh.read()


def subtasks_of_run(run_dir: str) -> list[str]:
    """The subtask names a run produced heuristics for."""
    if not os.path.isdir(run_dir):
        return []
    prefix, suffix = "best_heuristic_", ".py"
    return sorted(f[len(prefix):-len(suffix)] for f in os.listdir(run_dir)
                  if f.startswith(prefix) and f.endswith(suffix))


# ── per-task solvers ─────────────────────────────────────────────────────────

def _local_search_kwargs(instance, params: dict) -> dict:
    from atsp_utils import resolve_iter_limit
    return {
        "perturbation_moves": int(params.get("perturbation_moves", 30)),
        "iter_limit": resolve_iter_limit(params.get("iter_limit", 1000), instance.n),
        "time_limit": params.get("time_limit"),
    }


def _solve_gls(func, instance, params) -> float:
    from gls import guided_local_search
    # `func` is the update rule itself, not a matrix: MoH's GLS calls it once
    # per iteration. See the module docstring.
    _, cost = guided_local_search(instance.dist, func,
                                  **_local_search_kwargs(instance, params))
    return cost


def _solve_kgls(func, instance, params) -> float:
    from kgls import knowledge_guided_local_search
    _, cost = knowledge_guided_local_search(instance.dist, func,
                                            **_local_search_kwargs(instance, params))
    return cost


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


SOLVERS = {"atsp_gls": _solve_gls, "atsp_kgls": _solve_kgls,
           "atsp_constructive": _solve_constructive}


# ── driver ───────────────────────────────────────────────────────────────────

def run_benchmark(task: str, heuristic_code: str, instances,
                  params: dict | None = None, log=print) -> list[dict]:
    if task not in TASKS:
        raise ValueError(f"unknown MoH task {task!r}; expected {sorted(TASKS)}")
    func = load_callable(heuristic_code, TASKS[task]["entry"])
    params = default_params(task) if params is None else params
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
        except Exception as exc:      # a broken heuristic must not abort the sweep
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
        for key in ("framework", "task", "heuristic", "split", "subtask"):
            if meta.get(key):
                fh.write(f"- **{key}**: {meta[key]}\n")
        summary = payload["summary"]
        fh.write(f"\n**mean gap {summary['mean_gap_percent']:.3f}%** over "
                 f"{summary['n_solved']}/{summary['n_instances']} instances "
                 f"({summary['n_optimal']} optimal)\n\n")
        fh.write(markdown_table(records) + "\n")

    return {"csv": csv_path, "json": os.path.join(out_dir, f"{name}.json"),
            "markdown": md_path, "summary": payload["summary"]}
