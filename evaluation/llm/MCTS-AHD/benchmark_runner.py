"""Score an MCTS-AHD-designed ATSP heuristic on a set of instances.

Mirrors ``evaluation/llm/ReEvo/benchmark_runner.py`` and writes the same files,
so ``evaluation/llm/EoH/stats_analysis.py`` aggregates all three frameworks into
one table. Instances come from MCTS-AHD's own ``atsp/`` package, which is a copy
of ReEvo's, so the matrices and the reference costs are identical.

Budgets are **not** duplicated here: they are read from
``configs/llm/MCTS-AHD/cfg/evaluation/<task>.yaml``, the same file each task's
own ``eval.py`` reads, so a number produced by this runner and a number produced
by a run's post-hoc validation mean the same thing by construction rather than
by somebody remembering to keep two copies in step.
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
AHD = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")
for _p in (ROOT, AHD, os.path.join(AHD, "problems", "atsp_gls"),
           os.path.join(AHD, "problems", "atsp_kgls"),
           os.path.join(AHD, "problems", "atsp_aco")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.metrics import markdown_table, summarise  # noqa: E402
from atsp_utils import budget as task_budget  # noqa: E402

CSV_FIELDS = ["instance", "n", "cost", "ref_cost", "ref_kind", "gap_percent",
              "seconds", "status", "error"]

#: task -> the function name the LLM writes. The *budgets* are not here: they
#: come from configs/llm/MCTS-AHD/cfg/evaluation/<task>.yaml, the same file the
#: task's own eval.py reads, so a number produced by this runner and a number
#: produced by a run's post-hoc validation mean the same thing.
TASKS = {
    "atsp_constructive": {"entry": "select_next_node"},
    "atsp_gls": {"entry": "heuristics"},
    "atsp_kgls": {"entry": "arc_badness"},
    "atsp_aco": {"entry": "heuristics"},
}

#: One line per task, for the report headers.
TASK_SUMMARY = {
    "atsp_constructive": "greedy construction — the LLM designs the next-city rule",
    "atsp_gls": "guided local search — the LLM designs a static arc-badness guide matrix",
    "atsp_kgls": "knowledge-guided local search — the LLM designs a dynamic arc-badness rule",
    "atsp_aco": "ant colony optimisation — the LLM designs the arc desirability matrix",
}


def default_params(task: str, mood: str = "test") -> dict:
    """The engine budget for ``task``, read from the shared YAML config."""
    return task_budget(task, mood)


def load_callable(code: str, entry_point: str):
    """Compile a heuristic source string and return its entry point.

    MCTS-AHD names functions ``<name>``, ``<name>_v1``, ``<name>_v2`` ... during
    the search, so accept any of those and fall back to the bare name.
    """
    module = types.ModuleType("mcts_ahd_heuristic")
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
    path = os.path.join(AHD, "prompts", task, "seed_func.txt")
    with open(path, encoding="utf-8") as fh:
        return "import numpy as np\n\n" + fh.read().replace("_v1(", "(")


def load_heuristic_for_run(run_dir: str) -> tuple[str, str]:
    """Return ``(task, source_code)`` for a finished MCTS-AHD run directory."""
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

def _local_search_kwargs(instance, params: dict) -> dict:
    from atsp_utils import resolve_iter_limit
    return {
        "perturbation_moves": int(params.get("perturbation_moves", 30)),
        "iter_limit": resolve_iter_limit(params.get("iter_limit", 1000), instance.n),
        "time_limit": params.get("time_limit"),
    }


def _solve_gls(func, instance, params) -> float:
    from gls import guided_local_search
    dist = instance.dist
    guide = func(dist.copy())
    _, cost = guided_local_search(dist, guide, **_local_search_kwargs(instance, params))
    return cost


def _solve_kgls(func, instance, params) -> float:
    from kgls import knowledge_guided_local_search
    _, cost = knowledge_guided_local_search(instance.dist, func,
                                            **_local_search_kwargs(instance, params))
    return cost


def _solve_aco(func, instance, params) -> float:
    import time as _time
    from aco import ACO
    from atsp_utils import positive_matrix
    # No zeros in the matrix the heuristic sees, so 1/d stays finite; ~6% of
    # the arcs in the TSPLIB rbg instances cost exactly 0.
    aco = ACO(instance.dist, func(positive_matrix(instance.dist)),
              n_ants=int(params.get("n_ants", 20)),
              seed=int(params.get("seed", 2024)))
    time_limit = params.get("time_limit")
    deadline = None if time_limit is None else _time.perf_counter() + float(time_limit)
    return float(aco.run(int(params.get("n_iterations", 50)), deadline=deadline))


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
           "atsp_aco": _solve_aco, "atsp_constructive": _solve_constructive}


# ── driver ───────────────────────────────────────────────────────────────────

def run_benchmark(task: str, heuristic_code: str, instances,
                  params: dict | None = None, log=print) -> list[dict]:
    if task not in TASKS:
        raise ValueError(f"unknown MCTS-AHD task {task!r}; expected {sorted(TASKS)}")
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
