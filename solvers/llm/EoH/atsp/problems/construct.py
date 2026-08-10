"""Task 1 — constructive heuristic for ATSP (`select_next_node`)."""

from __future__ import annotations

from ..engines.construct import construct_cost
from .base import ATSP_CONTEXT, ATSPProblem


class ATSPConstruct(ATSPProblem):
    """EoH designs the next-city rule of a greedy ATSP tour construction."""

    task_name = "construct"
    entry_point = "select_next_node"

    template_program = '''
import numpy as np

def select_next_node(current_node: int, destination_node: int,
                     unvisited_nodes: np.ndarray,
                     distance_matrix: np.ndarray) -> int:
    """Select the next city to visit while constructing an ATSP tour.

    Args:
        current_node:     index of the city the salesman is currently at
        destination_node: index of the city the tour must finally return to
        unvisited_nodes:  1-D integer array of candidate cities not yet visited
        distance_matrix:  (n, n) directed cost matrix. distance_matrix[i, j] is
                          the cost of the arc i -> j and is generally different
                          from distance_matrix[j, i]. Read-only.
    Returns:
        next_node: one element of unvisited_nodes
    """
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])])
'''

    task_description = (
        ATSP_CONTEXT + " "
        "The tour is built step by step: starting from the depot, at each step one of the "
        "unvisited candidate cities is selected and appended to the partial tour. "
        "Design a novel algorithm to select the next city at each step. "
        "Useful signals include the outgoing costs distance_matrix[current_node][j], the "
        "incoming costs distance_matrix[j][current_node], the cost of eventually returning "
        "to the destination node, how isolated a candidate is with respect to the remaining "
        "unvisited cities, and the asymmetry distance_matrix[i][j] - distance_matrix[j][i], "
        "which indicates cities that are cheap to enter but expensive to leave. "
        "Avoid greedy choices that strand such cities until the end of the tour. "
        "The goal is to minimise the total cost of the closed tour."
    )

    def __init__(self, instances, n_candidates: int = 20, start_node: int = 0,
                 timeout: int = 60, n_processes: int = 1):
        super().__init__(instances, timeout=timeout, n_processes=n_processes)
        self.n_candidates = int(n_candidates)
        self.start_node = int(start_node)

    def solve_instance(self, callable_func, instance) -> float:
        return construct_cost(instance.dist, callable_func,
                              n_candidates=self.n_candidates,
                              start=self.start_node)

    def describe(self) -> dict:
        info = super().describe()
        info.update(n_candidates=self.n_candidates, start_node=self.start_node)
        return info
