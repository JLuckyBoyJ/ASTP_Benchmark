"""Ant Colony Optimisation for ATSP with a *directed* pheromone matrix.

On a symmetric TSP, ants deposit on both ``(u, v)`` and ``(v, u)``. That is
wrong for ATSP: traversing u -> v says nothing about the desirability of
v -> u. Here the pheromone matrix is genuinely directed and only the arc that
was actually traversed is reinforced. The LLM designs the update rule.
"""

from __future__ import annotations

import time

import numpy as np

from .tour import tour_cost

_EPS = 1e-12


def visibility(dist: np.ndarray) -> np.ndarray:
    """eta = 1 / d, direction-aware and safe for zero-cost arcs."""
    eta = 1.0 / (np.asarray(dist, dtype=np.float64) + _EPS)
    np.fill_diagonal(eta, 0.0)
    return eta


def construct_ant_tour(pheromone: np.ndarray, eta: np.ndarray, alpha: float,
                       beta: float, rng: np.random.Generator) -> np.ndarray:
    n = pheromone.shape[0]
    start = int(rng.integers(n))
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = np.empty(n, dtype=np.int64)
    tour[0] = start
    current = start
    for step in range(1, n):
        attract = (pheromone[current] ** alpha) * (eta[current] ** beta)
        attract = np.where(visited, 0.0, attract)
        total = float(attract.sum())
        if not np.isfinite(total) or total < 1e-300:
            attract = (~visited).astype(np.float64)
            total = float(attract.sum())
        nxt = int(rng.choice(n, p=attract / total))
        tour[step] = nxt
        visited[nxt] = True
        current = nxt
    return tour


def run_aco(dist: np.ndarray, update_fn, n_ants: int = 20, iter_max: int = 50,
            alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
            seed: int = 0, deadline: float | None = None) -> float:
    """Run ACO on one instance and return the best tour cost found."""
    n = int(dist.shape[0])
    eta = visibility(dist)
    pheromone = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(pheromone, 0.0)
    rng = np.random.default_rng(seed)

    best_tour = None
    best_cost = np.inf

    for iteration in range(iter_max):
        if deadline is not None and time.perf_counter() > deadline:
            break
        ant_tours, costs = [], []
        for _ in range(n_ants):
            tour = construct_ant_tour(pheromone, eta, alpha, beta, rng)
            cost = tour_cost(tour, dist)
            ant_tours.append(tour)
            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()

        pheromone = update_fn(
            pheromone.copy(), ant_tours, np.asarray(costs, dtype=np.float64),
            best_tour.copy(), float(best_cost), float(rho), iteration, iter_max,
        )
        pheromone = np.asarray(pheromone, dtype=np.float64)
        if pheromone.shape != (n, n) or not np.all(np.isfinite(pheromone)):
            raise ValueError("update_pheromone returned an invalid matrix")
        pheromone = np.maximum(pheromone, 1e-10)
        np.fill_diagonal(pheromone, 0.0)

    if best_tour is None:
        raise ValueError("ACO produced no tour")
    return float(best_cost)
