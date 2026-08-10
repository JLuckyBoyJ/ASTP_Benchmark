"""Task 2 — guided local search for ATSP (`update_edge_distance`).

This is the direct ATSP counterpart of the TSP-GLS experiment in the EoH paper
(Section 4, "Traveling salesman"): the LLM designs the rule that reshapes the
search landscape, while the local search itself is fixed. The only change is
that arcs are directed, so penalties are applied per arc rather than per edge,
and the local search uses Or-opt + swap instead of 2-opt (see engines/local_search.py).
"""

from __future__ import annotations

from ..engines.gls import guided_local_search
from ..engines.tour import candidate_lists
from .base import ATSP_CONTEXT, ATSPProblem


class ATSPGuidedLocalSearch(ATSPProblem):
    task_name = "gls"
    entry_point = "update_edge_distance"

    template_program = '''
import numpy as np

def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray,
                         edge_n_used: np.ndarray) -> np.ndarray:
    """Reshape the cost landscape so the local search escapes a local optimum.

    Args:
        edge_distance:  (n, n) directed cost matrix of the ATSP instance;
                        edge_distance[i, j] is the cost of the arc i -> j and
                        differs from edge_distance[j, i]
        local_opt_tour: 1-D array of length n giving the current local-optimal
                        tour as a visiting order (it closes back to its start)
        edge_n_used:    (n, n) counter of how many times each *directed* arc
                        i -> j has already been penalised in previous iterations
    Returns:
        updated_edge_distance: (n, n) matrix used by the next local search.
                               Arcs whose value is raised the most are the ones
                               the search will be pushed away from.
    """
    return edge_distance.copy()
'''

    task_description = (
        ATSP_CONTEXT + " "
        "A guided local search alternates between (a) an Or-opt and swap local search that "
        "preserves tour orientation, and (b) a perturbation step that modifies the cost "
        "matrix so the search is pushed out of the current local optimum. "
        "Given the original directed cost matrix, the current local-optimal tour, and the "
        "number of times each directed arc has already been penalised, design a novel "
        "strategy to update the cost matrix. "
        "Only the arcs whose updated cost is increased the most above their original cost "
        "are penalised and re-optimised, so your update effectively decides which arcs the "
        "search should abandon next. "
        "Remember that arc (i, j) and arc (j, i) are two different arcs with different costs "
        "and separate penalty counters: penalising one must not implicitly penalise the "
        "other. Consider arc cost, arc utility relative to the tour, how often an arc has "
        "already been penalised, and the asymmetry between the two directions. "
        "The goal is to minimise the final tour length."
    )

    def __init__(self, instances, time_limit: float = 10.0, ite_max: int = 1000,
                 perturbation_moves: int = 1, n_penalise: int = 5,
                 n_candidates: int = 10, max_seg: int = 3,
                 timeout: int = 120, n_processes: int = 1):
        super().__init__(instances, timeout=timeout, n_processes=n_processes)
        self.time_limit = float(time_limit)
        self.ite_max = int(ite_max)
        self.perturbation_moves = int(perturbation_moves)
        self.n_penalise = int(n_penalise)
        self.n_candidates = int(n_candidates)
        self.max_seg = int(max_seg)
        self._cand_cache: dict[str, object] = {}

    def _candidates(self, instance):
        cached = self._cand_cache.get(instance.name)
        if cached is None:
            cached = candidate_lists(instance.dist, self.n_candidates)
            self._cand_cache[instance.name] = cached
        return cached

    def solve_instance(self, callable_func, instance) -> float:
        _, cost, _ = guided_local_search(
            instance.dist, callable_func,
            time_limit=self.time_limit, ite_max=self.ite_max,
            perturbation_moves=self.perturbation_moves,
            n_penalise=self.n_penalise, n_candidates=self.n_candidates,
            max_seg=self.max_seg, cand=self._candidates(instance))
        return cost

    def describe(self) -> dict:
        info = super().describe()
        info.update(time_limit=self.time_limit, ite_max=self.ite_max,
                    perturbation_moves=self.perturbation_moves,
                    n_penalise=self.n_penalise, n_candidates=self.n_candidates,
                    max_seg=self.max_seg)
        return info
