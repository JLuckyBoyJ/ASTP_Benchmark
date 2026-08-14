"""Ant Colony Optimisation for ATSP, made asymmetric-safe.

Port of MCTS-AHD's own `problems/tsp_aco/aco.py`, byte-identical to the copy in
`solvers/llm/ReEvo/problems/atsp_aco/aco.py` so the two frameworks' ACO
heuristics are scored by the same engine. It keeps upstream's division of
labour: the
**pheromone update is fixed** (plain Ant System) and the LLM designs the
*heuristic matrix* — the prior desirability of each arc that multiplies the
pheromone in the transition rule. That is the mirror image of EoH's ACO task,
where the heuristic is fixed at 1/d and the LLM designs the pheromone update.

Two deviations from upstream:

1. **Directed pheromone.** Upstream deposits on both ``(u, v)`` and ``(v, u)``
   because TSP edges are undirected. For ATSP that is wrong — traversing
   u -> v says nothing about v -> u — so only the traversed arc is reinforced.
2. **NumPy instead of torch.** Upstream uses torch tensors; this port keeps the
   repo dependency-free. The algorithm (probabilistic construction, AS update,
   ``decay`` evaporation) is unchanged.
"""

from __future__ import annotations

import time

import numpy as np


class ACO:
    def __init__(self, distances: np.ndarray, heuristic: np.ndarray,
                 n_ants: int = 30, decay: float = 0.9,
                 alpha: float = 1.0, beta: float = 1.0, seed: int = 0):
        self.distances = np.asarray(distances, dtype=np.float64)
        self.problem_size = len(self.distances)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.pheromone = np.ones_like(self.distances)
        np.fill_diagonal(self.pheromone, 0.0)

        self.heuristic = np.asarray(heuristic, dtype=np.float64).copy()
        if self.heuristic.shape != self.distances.shape:
            raise ValueError("heuristics() must return a matrix shaped like the input")
        # Self-loops are never traversed, so whatever the heuristic says about
        # the diagonal is irrelevant — zero it before validating, otherwise a
        # perfectly good rule like 1/d trips the finiteness check.
        np.fill_diagonal(self.heuristic, 0.0)
        if not np.all(np.isfinite(self.heuristic)):
            raise ValueError("heuristics() returned a non-finite value")
        self.heuristic = np.maximum(self.heuristic, 1e-10)
        np.fill_diagonal(self.heuristic, 0.0)

        self.shortest_path = None
        self.lowest_cost = float("inf")
        self.rng = np.random.default_rng(seed)

    # ── construction ─────────────────────────────────────────────────────────

    def gen_path(self) -> np.ndarray:
        n = self.problem_size
        start = int(self.rng.integers(n))
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        path = np.empty(n, dtype=np.int64)
        path[0] = start
        current = start
        for step in range(1, n):
            weights = (self.pheromone[current] ** self.alpha) * \
                      (self.heuristic[current] ** self.beta)
            weights = np.where(visited, 0.0, weights)
            total = float(weights.sum())
            if not np.isfinite(total) or total < 1e-300:
                weights = (~visited).astype(np.float64)
                total = float(weights.sum())
            nxt = int(self.rng.choice(n, p=weights / total))
            path[step] = nxt
            visited[nxt] = True
            current = nxt
        return path

    def gen_path_cost(self, path: np.ndarray) -> float:
        return float(self.distances[path, np.roll(path, -1)].sum())

    # ── pheromone (fixed Ant System, directed) ───────────────────────────────

    def update_pheromone(self, paths, costs) -> None:
        self.pheromone *= self.decay
        for path, cost in zip(paths, costs):
            deposit = 1.0 / max(cost, 1e-12)
            # Only the arc actually traversed — never its reverse.
            self.pheromone[path, np.roll(path, -1)] += deposit
        np.fill_diagonal(self.pheromone, 0.0)

    # ── driver ───────────────────────────────────────────────────────────────

    def run(self, n_iterations: int, deadline: float | None = None) -> float:
        for _ in range(n_iterations):
            if deadline is not None and time.perf_counter() > deadline:
                break
            paths, costs = [], []
            for _ in range(self.n_ants):
                path = self.gen_path()
                cost = self.gen_path_cost(path)
                paths.append(path)
                costs.append(cost)
                if cost < self.lowest_cost:
                    self.lowest_cost = cost
                    self.shortest_path = path.copy()
            self.update_pheromone(paths, costs)
        return self.lowest_cost
