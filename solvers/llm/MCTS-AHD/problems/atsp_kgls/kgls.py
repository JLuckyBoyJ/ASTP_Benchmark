"""Knowledge-guided local search for ATSP.

KGLS (Arnold & Sörensen, *Knowledge-guided local search*, C&OR 2019) is guided
local search with three changes, all of which are here:

1. **A knowledge-based badness measure decides what to penalise.** Plain GLS
   penalises the tour edge with the largest ``cost / (1 + penalties)``. KGLS
   replaces the cost with a *badness* that mixes several properties of the edge
   — its length, how far it sits from the centre of the instance, how it
   compares with the cheapest alternatives at its endpoints. That measure is
   exactly the thing this task hands to the LLM to design.
2. **Moves are restricted to candidate neighbours**, so a penalisation costs a
   handful of delta evaluations rather than a full pass.
3. **Re-optimisation is local and sequential**: only the endpoints of the
   arc just penalised are re-examined, propagating outward only while moves
   keep improving.

How this differs from the ``atsp_gls`` task
-------------------------------------------
``atsp_gls`` asks for a **static** guide: one ``(n, n)`` matrix computed once
from the distance matrix, exactly as in ReEvo's ``tsp_gls``. This task asks for
a **dynamic** rule: ``arc_badness`` is called once per outer iteration and
receives the tour the search is currently sitting on and the penalty counters
accumulated so far, so it can react to what the search has already tried. That
is a strictly larger design space, and it is the one KGLS operates in.

Asymmetry
---------
2-opt is absent for the usual reason: reversing a segment flips every arc
inside it, so under ``d[i][j] != d[j][i]`` the move is neither O(1) to evaluate
nor cost-preserving. The local search is the Or-opt + swap search from
``atsp/engines``, shared with every other task in this repository. Penalties
are counted per **directed** arc: ``u -> v`` and ``v -> u`` have separate
counters.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from atsp.engines.local_search import local_search, local_search_around  # noqa: E402
from atsp.engines.tour import candidate_lists, nearest_neighbour_tour, tour_cost  # noqa: E402


def knowledge_guided_local_search(distmat: np.ndarray, badness_fn,
                                  perturbation_moves: int = 30,
                                  iter_limit: int = 1000,
                                  n_candidates: int = 10,
                                  max_seg: int = 3,
                                  time_limit: float | None = None,
                                  start: int = 0):
    """Return ``(best_tour, best_cost)``.

    Args:
        distmat: ``(n, n)`` directed cost matrix.
        badness_fn: the LLM-designed rule. Called once per outer iteration as
            ``badness_fn(distance_matrix, tour, penalty_count, iteration)`` and
            must return a length-``n`` array whose ``i``-th entry scores the
            tour arc ``tour[i] -> tour[(i + 1) % n]``. Larger means "abandon
            this arc sooner".
        perturbation_moves: penalise-and-reoptimise steps per outer iteration.
        iter_limit: outer iterations.
        time_limit: wall-clock cap in seconds. An expensive rule stops early
            rather than running forever; it is then judged on what it achieved
            inside the same budget every other rule gets.

    The badness vector is computed once per outer iteration and reused for that
    iteration's ``perturbation_moves`` penalisations, with the utility
    ``badness / (1 + penalty)`` recomputed each time from the live counters.
    Calling the rule on every single penalisation was measured to spend most of
    the budget inside the LLM's own NumPy code rather than in the search.
    """
    n = int(distmat.shape[0])
    deadline = None if time_limit is None else time.perf_counter() + float(time_limit)
    cand = candidate_lists(distmat, n_candidates)

    tour, cost = local_search(nearest_neighbour_tour(distmat, start), distmat,
                              cand, max_seg, deadline)
    best_tour, best_cost = list(tour), cost
    # Penalty weight, as in Voudouris-Tsang and in ReEvo's GLS port.
    k = 0.1 * best_cost / max(1, n)
    penalty = np.zeros((n, n), dtype=np.float64)
    current = list(best_tour)

    for iteration in range(iter_limit):
        if deadline is not None and time.perf_counter() > deadline:
            break

        tour_arr = np.asarray(current, dtype=np.int64)
        badness = np.asarray(badness_fn(distmat.copy(), tour_arr.copy(),
                                        penalty.copy(), iteration),
                             dtype=np.float64).ravel()
        if badness.shape != (n,):
            raise ValueError(
                f"arc_badness must return one value per tour arc, i.e. shape ({n},); "
                f"got {badness.shape}")
        if not np.all(np.isfinite(badness)):
            raise ValueError("arc_badness returned a non-finite value")
        # Ranking is scale-free, but negatives would invert the comparison
        # against the penalty denominator, so shift into the non-negative half.
        low = badness.min()
        if low < 0:
            badness = badness - low

        heads = tour_arr
        tails = np.roll(tour_arr, -1)

        moves = 0
        attempts = 0
        while moves < perturbation_moves and attempts < 4 * perturbation_moves:
            attempts += 1
            if deadline is not None and time.perf_counter() > deadline:
                break

            utility = badness / (1.0 + penalty[heads, tails])
            idx = int(np.argmax(utility))
            u, v = int(heads[idx]), int(tails[idx])

            penalty[u, v] += 1.0
            augmented = distmat + k * penalty
            current, delta = local_search_around(current, augmented, cand,
                                                 (u, v), max_seg)
            if delta < 0:
                moves += 1
                # The tour changed, so the cached arc list is stale; rebuild it
                # from the new tour but keep this iteration's badness values,
                # which are indexed by arc rather than by position.
                tour_arr = np.asarray(current, dtype=np.int64)
                new_heads = tour_arr
                new_tails = np.roll(tour_arr, -1)
                lookup = {(int(a), int(b)): badness[i]
                          for i, (a, b) in enumerate(zip(heads, tails))}
                default = float(np.median(badness))
                badness = np.array([lookup.get((int(a), int(b)), default)
                                    for a, b in zip(new_heads, new_tails)])
                heads, tails = new_heads, new_tails

        current, cost = local_search(current, distmat, cand, max_seg, deadline)
        if cost < best_cost - 1e-9:
            best_tour, best_cost = list(current), cost

    return best_tour, float(tour_cost(best_tour, distmat))


def solve(distmat: np.ndarray, badness_fn, **kwargs) -> float:
    """Convenience wrapper returning just the cost."""
    return knowledge_guided_local_search(distmat, badness_fn, **kwargs)[1]
