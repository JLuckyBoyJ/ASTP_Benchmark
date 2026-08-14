"""Guided Local Search for ATSP.

Structure follows the EoH paper's TSP-GLS experiment one-for-one; only the
moves and the penalty bookkeeping are made direction-aware:

  1. Nearest-neighbour start, then Or-opt + swap local search to a local optimum.
  2. Call the LLM-designed ``update_edge_distance`` to obtain an *augmented*
     distance matrix.
  3. Penalise the ``n_penalise`` most-augmented **directed** arcs (u -> v only;
     the reverse arc v -> u keeps its own penalty counter) and re-optimise
     locally around their endpoints under the augmented distances.
  4. Re-run local search on the *original* distances and keep the best tour.
  5. Every 50 iterations restart from the best-so-far tour.
"""

from __future__ import annotations

import time

import numpy as np

from .local_search import local_search, local_search_around
from .tour import candidate_lists, nearest_neighbour_tour, tour_cost


def guided_local_search(dist: np.ndarray, update_fn, time_limit: float = 10.0,
                        ite_max: int = 1000, perturbation_moves: int = 1,
                        n_penalise: int = 5, n_candidates: int = 10,
                        max_seg: int = 3, cand: np.ndarray | None = None,
                        init_tour=None):
    """Run GLS on one instance. Returns ``(best_tour, best_cost, iterations)``."""
    n = int(dist.shape[0])
    if cand is None:
        cand = candidate_lists(dist, n_candidates)
    deadline = time.perf_counter() + float(time_limit)

    tour = list(init_tour) if init_tour is not None else nearest_neighbour_tour(dist)
    tour, cost = local_search(tour, dist, cand, max_seg, deadline)
    best_tour, best_cost = list(tour), cost

    penalty = np.zeros((n, n), dtype=np.float64)
    iteration = 0

    while iteration < ite_max and time.perf_counter() < deadline:
        for _ in range(perturbation_moves):
            augmented = update_fn(dist.copy(), np.asarray(tour, dtype=np.int64),
                                  penalty.copy())
            augmented = np.asarray(augmented, dtype=np.float64)
            if augmented.shape != (n, n) or not np.all(np.isfinite(augmented)):
                raise ValueError("update_edge_distance returned an invalid matrix")

            surplus = augmented - dist
            np.fill_diagonal(surplus, -np.inf)

            for _ in range(n_penalise):
                flat = int(np.argmax(surplus))
                u, v = divmod(flat, n)
                if not np.isfinite(surplus[u, v]) or surplus[u, v] <= 0:
                    break
                penalty[u, v] += 1.0
                surplus[u, v] = -np.inf
                tour, _ = local_search_around(tour, augmented, cand, (u, v), max_seg)

            if time.perf_counter() >= deadline:
                break

        tour, cost = local_search(tour, dist, cand, max_seg, deadline)
        if cost < best_cost - 1e-9:
            best_tour, best_cost = list(tour), cost

        iteration += 1
        if iteration % 50 == 0:
            tour = list(best_tour)

    return best_tour, float(best_cost), iteration


def solve_instance_gls(dist: np.ndarray, ref_cost: float | None, update_fn,
                       time_limit: float = 10.0, ite_max: int = 1000,
                       perturbation_moves: int = 1, n_penalise: int = 5,
                       n_candidates: int = 10, max_seg: int = 3,
                       cand: np.ndarray | None = None) -> float:
    """GLS wrapper returning the tour cost (raises on a broken heuristic)."""
    _, cost, _ = guided_local_search(
        dist, update_fn, time_limit=time_limit, ite_max=ite_max,
        perturbation_moves=perturbation_moves, n_penalise=n_penalise,
        n_candidates=n_candidates, max_seg=max_seg, cand=cand)
    return cost


__all__ = ["guided_local_search", "solve_instance_gls", "tour_cost"]
