"""Guided local search for ATSP — ReEvo's variant, made asymmetric-safe.

This is a port of ReEvo's `problems/tsp_gls/gls.py`, keeping its algorithm:

* the LLM supplies a **static** guide matrix (prior "badness" of each arc),
  computed once from the distance matrix — unlike EoH, whose heuristic is
  called every iteration;
* the perturbation repeatedly penalises the tour arc with the highest utility
  ``guide[u, v] / (1 + penalty[u, v])``, then re-optimises locally around that
  arc's endpoints under ``distmat + k * penalty``;
* ``k = 0.1 * initial_cost / n``, as upstream.

Two deviations, both forced by the problem rather than by taste:

1. **No 2-opt.** Upstream's local search is 2-opt + relocate, and 2-opt
   reverses a tour segment — under an asymmetric matrix every arc inside the
   reversed segment flips direction, so the move is neither O(1) to evaluate
   nor cost-preserving. We use the Or-opt + swap search from
   this solver's own ``atsp/engines``.
2. **No numba.** Upstream JIT-compiles with explicit float32 signatures; this
   port is plain NumPy so the repo has no compiler dependency. It is slower,
   which is why the budget is wall-clock bounded.

Penalties are applied to the directed arc ``u -> v`` only; ``v -> u`` keeps its
own counter.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from atsp.engines.local_search import local_search, local_search_around
from atsp.engines.tour import (
    candidate_lists,
    nearest_neighbour_tour,
    tour_cost,
)


def guided_local_search(distmat: np.ndarray, guide: np.ndarray,
                        perturbation_moves: int = 30, iter_limit: int = 1000,
                        n_candidates: int = 10, max_seg: int = 3,
                        time_limit: float | None = None,
                        start: int = 0):
    """Return ``(best_tour, best_cost)``.

    Args:
        distmat: (n, n) directed cost matrix.
        guide:   (n, n) prior badness of each arc, as designed by the LLM.
                 Only its *relative* values matter — it selects which arc to
                 penalise, it is never added to the cost.
        perturbation_moves: penalise/re-optimise steps per iteration.
        iter_limit: outer iterations.
        time_limit: wall-clock cap in seconds; the search stops early rather
                    than letting an expensive guide run forever.
    """
    n = int(distmat.shape[0])
    guide = np.asarray(guide, dtype=np.float64)
    if guide.shape != (n, n) or not np.all(np.isfinite(guide)):
        raise ValueError("heuristics() must return a finite (n, n) matrix")

    deadline = None if time_limit is None else time.perf_counter() + float(time_limit)
    cand = candidate_lists(distmat, n_candidates)

    tour, cost = local_search(nearest_neighbour_tour(distmat, start), distmat,
                              cand, max_seg, deadline)
    best_tour, best_cost = list(tour), cost
    k = 0.1 * best_cost / max(1, n)
    penalty = np.zeros((n, n), dtype=np.float64)
    current = list(best_tour)

    for _ in range(iter_limit):
        if deadline is not None and time.perf_counter() > deadline:
            break

        # ── perturbation ──────────────────────────────────────────────────
        moves = 0
        attempts = 0
        while moves < perturbation_moves and attempts < 4 * perturbation_moves:
            attempts += 1
            if deadline is not None and time.perf_counter() > deadline:
                break

            # the tour arc with the highest utility is the one to abandon
            u = v = -1
            best_util = -np.inf
            for i in range(n):
                a, b = current[i], current[(i + 1) % n]
                util = guide[a, b] / (1.0 + penalty[a, b])
                if util > best_util:
                    best_util, u, v = util, a, b
            if u < 0:
                break

            penalty[u, v] += 1.0
            augmented = distmat + k * penalty
            current, delta = local_search_around(current, augmented, cand,
                                                 (u, v), max_seg)
            if delta < 0:
                moves += 1

        # ── re-optimise on the true costs and keep the best ───────────────
        current, cost = local_search(current, distmat, cand, max_seg, deadline)
        if cost < best_cost - 1e-9:
            best_tour, best_cost = list(current), cost

    return best_tour, float(tour_cost(best_tour, distmat))


def solve(distmat: np.ndarray, guide: np.ndarray, **kwargs) -> float:
    """Convenience wrapper returning just the cost."""
    return guided_local_search(distmat, guide, **kwargs)[1]
