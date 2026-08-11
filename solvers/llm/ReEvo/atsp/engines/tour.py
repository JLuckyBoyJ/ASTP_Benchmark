"""Tour primitives for the asymmetric TSP.

A tour is a Python list (or 1-D int array) of length ``n`` containing every
node exactly once. The tour is closed implicitly: the successor of the last
node is the first node. Every routine is direction-aware because
``D[i, j] != D[j, i]`` in general.
"""

from __future__ import annotations

import numpy as np


def tour_cost(tour, dist: np.ndarray) -> float:
    """Closed-tour cost of ``tour`` under the directed matrix ``dist``."""
    t = np.asarray(tour, dtype=np.int64)
    return float(dist[t, np.roll(t, -1)].sum())


def is_valid_tour(tour, n: int) -> bool:
    """True when ``tour`` is a permutation of ``range(n)``."""
    t = np.asarray(tour, dtype=np.int64).ravel()
    if t.size != n:
        return False
    return bool(np.array_equal(np.sort(t), np.arange(n)))


def nearest_neighbour_tour(dist: np.ndarray, start: int = 0) -> list[int]:
    """Greedy nearest-neighbour tour following *outgoing* arcs."""
    n = dist.shape[0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start] = False
    tour = [int(start)]
    current = int(start)
    for _ in range(n - 1):
        row = np.where(unvisited, dist[current], np.inf)
        nxt = int(np.argmin(row))
        tour.append(nxt)
        unvisited[nxt] = False
        current = nxt
    return tour


def greedy_insertion_tour(dist: np.ndarray, start: int = 0) -> list[int]:
    """Cheapest-insertion tour (a second, structurally different start point)."""
    n = dist.shape[0]
    remaining = [i for i in range(n) if i != start]
    # seed with the cheapest 2-cycle out of `start`
    first = min(remaining, key=lambda j: dist[start, j] + dist[j, start])
    tour = [start, first]
    remaining.remove(first)
    while remaining:
        best = (np.inf, None, None)
        for node in remaining:
            for pos in range(len(tour)):
                a = tour[pos]
                b = tour[(pos + 1) % len(tour)]
                delta = dist[a, node] + dist[node, b] - dist[a, b]
                if delta < best[0]:
                    best = (delta, node, pos + 1)
        _, node, pos = best
        tour.insert(pos, node)
        remaining.remove(node)
    return [int(v) for v in tour]


def candidate_lists(dist: np.ndarray, k: int = 10) -> np.ndarray:
    """For every node, the ``k`` most attractive partners.

    An arc between ``i`` and ``j`` is worth trying if it is cheap in *either*
    direction, so the ranking uses ``min(D[i, j], D[j, i])``. Self-loops are
    excluded. Returns an ``(n, k)`` int array.
    """
    n = dist.shape[0]
    k = int(max(1, min(k, n - 1)))
    score = np.minimum(dist, dist.T).astype(np.float64, copy=True)
    np.fill_diagonal(score, np.inf)
    order = np.argsort(score, axis=1, kind="stable")
    return np.ascontiguousarray(order[:, :k].astype(np.int64))


def tour_edges(tour) -> list[tuple[int, int]]:
    """Directed arcs (u, v) of the closed tour."""
    t = [int(v) for v in tour]
    return [(t[i], t[(i + 1) % len(t)]) for i in range(len(t))]
