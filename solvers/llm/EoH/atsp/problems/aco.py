"""Task 3 — pheromone update rule for ACO on ATSP (`update_pheromone`)."""

from __future__ import annotations

import numpy as np

from ..engines.aco import run_aco
from .base import ATSP_CONTEXT, ATSPProblem


class ATSPAntColony(ATSPProblem):
    task_name = "aco"
    entry_point = "update_pheromone"

    template_program = '''
import numpy as np

def update_pheromone(pheromone: np.ndarray, ant_tours: list, tour_costs: np.ndarray,
                     best_tour: np.ndarray, best_cost: float,
                     rho: float, iteration: int, max_iterations: int) -> np.ndarray:
    """Update the directed pheromone matrix after one ACO iteration.

    Args:
        pheromone:      (n, n) DIRECTED pheromone matrix; pheromone[i, j] refers
                        to the arc i -> j only
        ant_tours:      list of m arrays of shape (n,), each a city visiting
                        order produced by one ant this iteration
        tour_costs:     (m,) tour cost of each ant (lower is better)
        best_tour:      (n,) best tour found so far across all iterations
        best_cost:      cost of best_tour
        rho:            evaporation rate in (0, 1)
        iteration:      current iteration index (0-based)
        max_iterations: total number of ACO iterations planned
    Returns:
        updated (n, n) pheromone matrix with non-negative entries
    """
    n = pheromone.shape[0]
    pheromone = (1.0 - rho) * pheromone
    for tour, cost in zip(ant_tours, tour_costs):
        deposit = 1.0 / cost
        for i in range(n):
            u, v = int(tour[i]), int(tour[(i + 1) % n])
            pheromone[u, v] += deposit
    return pheromone
'''

    task_description = (
        ATSP_CONTEXT + " "
        "Ant Colony Optimisation builds tours probabilistically: an ant in city i moves to "
        "city j with probability proportional to pheromone[i, j]**alpha * (1 / d[i, j])**beta. "
        "After every iteration the pheromone matrix is updated to reinforce promising arcs "
        "and evaporate the rest. Design a novel pheromone update rule. "
        "Classic strategies you may draw on or depart from include Ant System (every ant "
        "deposits proportionally to 1/cost), Elitist Ant System (bonus deposit on the "
        "best-so-far tour), MAX-MIN Ant System (only the best ant deposits and pheromone is "
        "clamped to [tau_min, tau_max]), and rank-based Ant System (the top ants deposit with "
        "linearly decreasing weights); adaptive rules that shift from exploration to "
        "exploitation as the iteration counter grows are also allowed. "
        "Crucially, the pheromone matrix is DIRECTED: traversing i -> j says nothing about the "
        "desirability of j -> i, so deposit on the traversed arc only and never symmetrise "
        "the matrix. The goal is to minimise the average best tour cost found."
    )

    def __init__(self, instances, n_ants: int = 20, iter_max: int = 50,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
                 n_runs: int = 1, seed: int = 2024, timeout: int = 120,
                 n_processes: int = 1):
        super().__init__(instances, timeout=timeout, n_processes=n_processes)
        self.n_ants = int(n_ants)
        self.iter_max = int(iter_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.n_runs = int(n_runs)
        self.seed = int(seed)

    def solve_instance(self, callable_func, instance) -> float:
        costs = [
            run_aco(instance.dist, callable_func, n_ants=self.n_ants,
                    iter_max=self.iter_max, alpha=self.alpha, beta=self.beta,
                    rho=self.rho, seed=self.seed + run)
            for run in range(self.n_runs)
        ]
        return float(np.mean(costs))

    def describe(self) -> dict:
        info = super().describe()
        info.update(n_ants=self.n_ants, iter_max=self.iter_max, alpha=self.alpha,
                    beta=self.beta, rho=self.rho, n_runs=self.n_runs, seed=self.seed)
        return info
