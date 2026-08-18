"""Ruin-and-recreate for ATSP.

Loop: destroy (LLM-designed node selection) -> cheapest insertion -> Or-opt/swap
local search -> accept if improved. The insertion delta
``d[a, x] + d[x, b] - d[a, b]`` is already direction-aware; the local search is
the asymmetric-safe one from ``local_search.py``.
"""

from __future__ import annotations

import time

import numpy as np

from .local_search import local_search
from .tour import candidate_lists, nearest_neighbour_tour, tour_cost


def cheapest_insertion(partial: list[int], removed: list[int],
                       dist: np.ndarray) -> list[int]:
    tour = list(partial)
    for node in removed:
        best_delta = np.inf
        best_pos = 1
        m = len(tour)
        for i in range(m):
            a = tour[i]
            b = tour[(i + 1) % m]
            delta = dist[a, node] + dist[node, b] - dist[a, b]
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1
        tour.insert(best_pos, int(node))
    return tour


def run_rnr(dist: np.ndarray, destroy_fn, n_destroy: int | None = None,
            iter_max: int = 100, time_limit: float = 5.0,
            n_candidates: int = 10, max_seg: int = 3,
            cand: np.ndarray | None = None) -> float:
    """Run ruin-and-recreate on one instance; return the best tour cost."""
    n = int(dist.shape[0])
    if cand is None:
        cand = candidate_lists(dist, n_candidates)
    if n_destroy is None:
        n_destroy = max(2, n // 5)
    n_destroy = int(min(n_destroy, n - 3))
    deadline = time.perf_counter() + float(time_limit)

    tour = nearest_neighbour_tour(dist)
    tour, cost = local_search(tour, dist, cand, max_seg, deadline)
    best_tour, best_cost = list(tour), cost

    dist_ro = dist.view()
    dist_ro.flags.writeable = False

    for _ in range(iter_max):
        if time.perf_counter() > deadline:
            break

        current = np.asarray(best_tour, dtype=np.int64)
        selected = destroy_fn(current.copy(), dist_ro, n_destroy)
        selected = np.asarray(selected, dtype=np.int64).ravel()
        if selected.size < n_destroy:
            continue
        selected = selected[:n_destroy]
        if selected.min() < 0 or selected.max() >= n:
            raise ValueError("destroy_nodes returned an out-of-range node")
        removed = list(dict.fromkeys(int(v) for v in selected))

        removed_set = set(removed)
        partial = [int(v) for v in best_tour if v not in removed_set]
        if len(partial) < 2:
            continue

        candidate_tour = cheapest_insertion(partial, removed, dist)
        candidate_tour, cost = local_search(candidate_tour, dist, cand, max_seg, deadline)
        if cost < best_cost - 1e-9:
            best_tour, best_cost = list(candidate_tour), cost

    return float(tour_cost(best_tour, dist))
