"""Greedy constructive engine for ATSP.

The LLM-designed ``select_next_node`` picks, at every step, which unvisited node
to move to. Candidates are restricted to the ``n_candidates`` cheapest
*outgoing* arcs from the current node — the asymmetric analogue of the nearest
neighbour list used in the original EoH TSP experiment.
"""

from __future__ import annotations

import numpy as np

from .tour import tour_cost


def _readonly(matrix: np.ndarray) -> np.ndarray:
    view = matrix.view()
    view.flags.writeable = False
    return view


def greedy_construct(dist: np.ndarray, select_fn, n_candidates: int = 20,
                     start: int = 0) -> list[int]:
    """Build a tour by repeatedly calling ``select_fn``.

    Raises ValueError if the heuristic returns an invalid node, which the caller
    turns into a failed evaluation (fitness ``None``) — same contract as EoH.
    """
    n = int(dist.shape[0])
    dist_ro = _readonly(dist)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    route = [int(start)]
    current = int(start)
    destination = int(start)

    for _ in range(n - 1):
        unvisited = np.flatnonzero(~visited)
        if unvisited.size > n_candidates:
            order = np.argsort(dist[current, unvisited], kind="stable")
            unvisited = unvisited[order[:n_candidates]]
        choice = select_fn(current, destination, unvisited.copy(), dist_ro)
        choice = int(np.asarray(choice).ravel()[0])
        if choice < 0 or choice >= n or visited[choice]:
            raise ValueError(f"select_next_node returned an invalid node: {choice}")
        route.append(choice)
        visited[choice] = True
        current = choice

    return route


def construct_cost(dist: np.ndarray, select_fn, n_candidates: int = 20,
                   start: int = 0) -> float:
    return tour_cost(greedy_construct(dist, select_fn, n_candidates, start), dist)
