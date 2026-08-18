"""Synthetic ATSP instance families used for heuristic *evolution*.

The EoH paper evolves TSP heuristics on 64 randomly generated TSP100 instances
and reports the average gap to the optimum as the fitness. We keep that
protocol and only swap the instance family for asymmetric ones:

``uniform``
    ``D[i, j] ~ U{1, ..., max_cost}`` drawn independently for each direction —
    the classical random ATSP benchmark, maximally asymmetric.

``asymmetric_clustered``
    Clustered 2-D locations give a Euclidean base cost, which is then distorted
    by a per-node "elevation" so that travelling i -> j costs more than
    j -> i whenever j is higher. This produces structured, realistic asymmetry
    (one-way streets, gradients, wind) instead of pure noise.

Reference costs
---------------
Fitness is an optimality gap, so every training instance needs a reference
cost. TSPLIB instances come with proven optima; synthetic ones are scored
against the best tour found by :func:`reference_cost`, a deterministic
multi-start Or-opt/swap local search. Instances therefore carry
``ref_kind="heuristic"`` and every report states this explicitly.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .instance import ATSPInstance
from ..engines.gls import guided_local_search
from ..engines.local_search import local_search
from ..engines.rnr import cheapest_insertion
from ..engines.tour import (
    candidate_lists,
    greedy_insertion_tour,
    nearest_neighbour_tour,
    tour_cost,
)

FAMILIES = ("uniform", "asymmetric_clustered", "scheduling_constrained",
            "stacker_crane")


# ── generators ────────────────────────────────────────────────────────────────

def _generate_uniform(n: int, rng: np.random.Generator, max_cost: int = 1000) -> np.ndarray:
    dist = rng.integers(1, max_cost + 1, size=(n, n)).astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    return dist


def _generate_asymmetric_clustered(n: int, rng: np.random.Generator,
                                   n_clusters: int | None = None,
                                   asymmetry: float | None = None,
                                   scale: float | None = None) -> np.ndarray:
    if n_clusters is None:
        n_clusters = max(2, int(round(np.sqrt(n) / 2)))
    if asymmetry is None:
        asymmetry = float(rng.uniform(0.2, 0.8))
    if scale is None:
        scale = float(10 ** rng.uniform(2.0, 4.0))

    centres = rng.random((n_clusters, 2))
    assign = rng.integers(0, n_clusters, size=n)
    coords = np.clip(centres[assign] + rng.normal(0.0, 0.08, size=(n, 2)), 0.0, 1.0)

    base = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2) * scale
    elevation = rng.random(n)
    slope = elevation[None, :] - elevation[:, None]          # + when going "uphill"
    dist = base * (1.0 + asymmetry * slope)
    dist = np.maximum(dist, 1.0)
    dist = np.round(dist)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def _generate_scheduling_constrained(n: int, rng: np.random.Generator,
                                     scale: float = 1000.0) -> np.ndarray:
    """Simulate machine setup times with stage precedence constraints and asymmetric transitions (matching TSPLIB rbg instances)."""
    base_setup = rng.integers(1, int(scale / 10), size=(n, n)).astype(np.float64)
    # Stage-based sequence constraints
    n_stages = max(2, min(6, n // 10))
    stages = rng.integers(0, n_stages, size=n)
    penalty_mask = stages[:, None] > stages[None, :]
    base_setup[penalty_mask] *= rng.uniform(5.0, 20.0)
    np.fill_diagonal(base_setup, 0.0)
    return np.round(base_setup).astype(np.float64)


def _generate_stacker_crane(n: int, rng: np.random.Generator,
                            n_cells: int = 600, aspect: int = 3,
                            lift_speed: float = 3.0, hot: int = 8,
                            hot_share: float = 0.75,
                            min_move: float = 0.25) -> np.ndarray:
    """The regime the TSPLIB `rbg` instances live in, which nothing else here covers.

    Those are stacker-crane schedules: a request runs from a pickup cell to a
    drop-off cell in a rack, and the cost of following request i with request j
    is the *empty* travel from i's drop-off to j's pickup. The crane drives and
    lifts at once, so that travel is a Chebyshev distance; a fixed minimum
    manoeuvre time floors it; and it is exactly zero when i ends where j starts.

    Measured against rbg323/358/403/443 this reproduces all the properties the
    other families miss — 33 distinct integer costs, 6% zero-cost arcs, 77% of
    rows holding a free onward arc, near-zero directional correlation, and a
    cost range only 2x the median. Those properties are not cosmetic: on a
    matrix this flat, a guide proportional to arc cost carries almost no
    information, and a constant guide beats it by 1.2 points at n>=323.
    """
    width = max(2, int(round(np.sqrt(n_cells * aspect))))
    height = max(2, int(round(n_cells / width)))
    cells = np.array([(x, y) for x in range(width) for y in range(height)], float)
    hotset = cells[rng.choice(len(cells), size=min(hot, len(cells)), replace=False)]

    def positions():
        """Most requests touch a few busy cells; the rest are scattered."""
        busy = rng.random(n) < hot_share
        return np.where(busy[:, None],
                        hotset[rng.integers(0, len(hotset), n)],
                        cells[rng.integers(0, len(cells), n)])

    pickup, dropoff = positions(), positions()
    travel = np.maximum(
        np.abs(dropoff[:, 0][:, None] - pickup[:, 0][None, :]),
        np.abs(dropoff[:, 1][:, None] - pickup[:, 1][None, :]) * lift_speed)
    dist = np.where(travel == 0, 0.0, np.maximum(travel, min_move * travel.max()))
    np.fill_diagonal(dist, 0.0)
    return np.round(dist).astype(np.float64)


def generate_instance(family: str, n: int, seed: int, **kwargs) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "uniform":
        return _generate_uniform(n, rng, **kwargs)
    if family == "asymmetric_clustered":
        return _generate_asymmetric_clustered(n, rng, **kwargs)
    if family == "scheduling_constrained":
        return _generate_scheduling_constrained(n, rng, **kwargs)
    if family == "stacker_crane":
        return _generate_stacker_crane(n, rng, **kwargs)
    raise ValueError(f"Unknown ATSP family {family!r}; expected one of {FAMILIES}")


# ── reference solver ──────────────────────────────────────────────────────────

def _classic_gls_penalty(edge_distance: np.ndarray, local_opt_tour: np.ndarray,
                         edge_n_used: np.ndarray) -> np.ndarray:
    """Voudouris & Tsang penalty on directed arcs — the reference solver's guide."""
    updated = edge_distance.copy()
    n = len(local_opt_tour)
    for k in range(n):
        u = int(local_opt_tour[k])
        v = int(local_opt_tour[(k + 1) % n])
        updated[u, v] += edge_distance[u, v] / (1.0 + edge_n_used[u, v])
    return updated


#: (multi-starts, GLS iterations, ruin-and-recreate perturbations, alternations)
EFFORT_LEVELS = {
    "low": (4, 200, 200, 1),
    "medium": (8, 500, 600, 3),
    "high": (12, 1000, 1500, 5),
}


def reference_cost(dist: np.ndarray, effort: str = "medium", seed: int = 2024,
                   n_candidates: int = 12, time_cap: float = 600.0,
                   n_starts: int | None = None, gls_iterations: int | None = None,
                   n_perturb: int | None = None, rounds: int | None = None) -> float:
    """Deterministic strong-baseline cost used as the optimality reference.

    Phase 1 multi-start construction (nearest neighbour from several depots plus
            cheapest insertion), each polished with Or-opt/swap local search.
    Phase 2 alternating rounds of guided local search (classic Voudouris-Tsang
            penalty) and random ruin-and-recreate, always restarting from the
            incumbent best.

    The budget is expressed in *iterations*, not wall-clock time, so the same
    matrix yields the same reference on any machine (``time_cap`` is only a
    safety net). The reference is deliberately far stronger than what a task
    achieves inside its per-evaluation budget, so training gaps stay positive
    and comparable across tasks; ``effort="low"`` trades that guarantee for
    speed on large instances.
    """
    if effort not in EFFORT_LEVELS:
        raise ValueError(f"effort must be one of {sorted(EFFORT_LEVELS)}")
    d_starts, d_gls, d_perturb, d_rounds = EFFORT_LEVELS[effort]
    n_starts = d_starts if n_starts is None else n_starts
    gls_iterations = d_gls if gls_iterations is None else gls_iterations
    n_perturb = d_perturb if n_perturb is None else n_perturb
    rounds = d_rounds if rounds is None else rounds

    n = int(dist.shape[0])
    cand = candidate_lists(dist, n_candidates)
    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + time_cap

    # ── phase 1: multi-start ─────────────────────────────────────────────────
    step = max(1, n // max(1, n_starts))
    best_tour, best = None, np.inf
    for start in list(range(0, n, step))[:n_starts]:
        tour, cost = local_search(nearest_neighbour_tour(dist, start), dist, cand)
        if cost < best:
            best_tour, best = tour, cost
    tour, cost = local_search(greedy_insertion_tour(dist), dist, cand)
    if cost < best:
        best_tour, best = tour, cost

    # ── phase 2: GLS <-> ruin-and-recreate ───────────────────────────────────
    n_destroy = max(2, min(n - 3, n // 5))
    for _ in range(max(1, rounds)):
        if time.perf_counter() > deadline:
            break

        if gls_iterations > 0:
            tour, cost, _ = guided_local_search(
                dist, _classic_gls_penalty,
                time_limit=max(1.0, deadline - time.perf_counter()),
                ite_max=gls_iterations, n_candidates=n_candidates,
                cand=cand, init_tour=best_tour)
            if cost < best - 1e-9:
                best_tour, best = tour, cost

        for _ in range(n_perturb):
            if time.perf_counter() > deadline:
                break
            removed = [int(v) for v in rng.choice(n, size=n_destroy, replace=False)]
            removed_set = set(removed)
            partial = [int(v) for v in best_tour if v not in removed_set]
            if len(partial) < 2:
                continue
            tour, cost = local_search(cheapest_insertion(partial, removed, dist),
                                      dist, cand)
            if cost < best - 1e-9:
                best_tour, best = tour, cost

    return float(tour_cost(best_tour, dist))


# ── datasets ──────────────────────────────────────────────────────────────────

def dataset_filename(family: str, n: int, count: int, seed: int) -> str:
    return f"atsp_{family}_n{n}_c{count}_s{seed}.npz"


def generate_dataset(family: str, n: int, count: int, seed: int = 2024,
                     with_reference: bool = True, effort: str = "medium",
                     progress=None) -> list[ATSPInstance]:
    """Generate ``count`` instances of ``family`` with size ``n``."""
    instances: list[ATSPInstance] = []
    for idx in range(count):
        inst_seed = seed * 100_000 + idx
        dist = generate_instance(family, n, inst_seed)
        ref = (reference_cost(dist, effort=effort, seed=inst_seed)
               if with_reference else None)
        instances.append(ATSPInstance(
            name=f"{family}_n{n}_{idx:03d}",
            dist=dist,
            ref_cost=ref,
            ref_kind="heuristic" if ref is not None else "unknown",
            source="synthetic",
            meta={"family": family, "seed": inst_seed, "index": idx},
        ))
        if progress is not None:
            progress(idx + 1, count, instances[-1])
    return instances


def save_dataset(path: str, instances: list[ATSPInstance]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        dist=np.stack([ins.dist for ins in instances]),
        ref_cost=np.asarray([np.nan if ins.ref_cost is None else ins.ref_cost
                             for ins in instances], dtype=np.float64),
        names=np.asarray([ins.name for ins in instances]),
        meta=np.asarray(json.dumps([ins.meta for ins in instances])),
        ref_kind=np.asarray([ins.ref_kind for ins in instances]),
    )
    return path


def load_dataset(path: str) -> list[ATSPInstance]:
    with np.load(path, allow_pickle=False) as blob:
        dists = blob["dist"]
        refs = blob["ref_cost"]
        names = blob["names"]
        kinds = blob["ref_kind"] if "ref_kind" in blob else None
        metas = json.loads(str(blob["meta"])) if "meta" in blob else [{}] * len(names)
    out = []
    for idx, name in enumerate(names):
        ref = float(refs[idx])
        out.append(ATSPInstance(
            name=str(name),
            dist=np.ascontiguousarray(dists[idx], dtype=np.float64),
            ref_cost=None if np.isnan(ref) else ref,
            ref_kind=str(kinds[idx]) if kinds is not None else "heuristic",
            source="synthetic",
            meta=metas[idx] if idx < len(metas) else {},
        ))
    return out
