"""Task 4 — destroy operator for ruin-and-recreate on ATSP (`destroy_nodes`)."""

from __future__ import annotations

from ..engines.rnr import run_rnr
from ..engines.tour import candidate_lists
from .base import ATSP_CONTEXT, ATSPProblem


class ATSPRuinAndRecreate(ATSPProblem):
    task_name = "rnr"
    entry_point = "destroy_nodes"

    template_program = '''
import numpy as np

def destroy_nodes(current_tour: np.ndarray, distance_matrix: np.ndarray,
                  n_destroy: int) -> np.ndarray:
    """Choose which cities to rip out of the current ATSP tour (ruin phase).

    Args:
        current_tour:    1-D integer array of length n giving the current tour
                         as a visiting order (it closes back to its start)
        distance_matrix: (n, n) directed cost matrix; distance_matrix[i, j] is
                         the cost of the arc i -> j and differs from
                         distance_matrix[j, i]. Read-only.
        n_destroy:       exact number of cities to remove
    Returns:
        nodes_to_remove: integer array of n_destroy distinct city indices
    """
    return np.random.choice(current_tour, size=n_destroy, replace=False)
'''

    task_description = (
        ATSP_CONTEXT + " "
        "A ruin-and-recreate search repeatedly removes a set of cities from the incumbent "
        "tour (ruin), reinserts them one by one at their cheapest position (recreate), and "
        "then polishes the tour with an Or-opt and swap local search. "
        "Design a novel destroy operator that decides which cities to remove. "
        "The quality of the operator lies in the trade-off between targeting badly placed "
        "cities and keeping enough diversity to escape local optima; removing a spatially or "
        "structurally related group is usually more effective than removing unrelated ones. "
        "For ATSP, note that a city can be cheap to enter but expensive to leave, so the cost "
        "an individual city contributes is the sum of its incoming arc and its outgoing arc, "
        "and relatedness between two cities should account for both directions. "
        "The operator must return exactly n_destroy distinct valid city indices. "
        "The goal is to minimise the final best tour cost."
    )

    def __init__(self, instances, n_destroy: int | None = None, iter_max: int = 100,
                 time_limit: float = 5.0, n_candidates: int = 10, max_seg: int = 3,
                 timeout: int = 120, n_processes: int = 1):
        super().__init__(instances, timeout=timeout, n_processes=n_processes)
        self.n_destroy = n_destroy
        self.iter_max = int(iter_max)
        self.time_limit = float(time_limit)
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
        return run_rnr(instance.dist, callable_func, n_destroy=self.n_destroy,
                       iter_max=self.iter_max, time_limit=self.time_limit,
                       n_candidates=self.n_candidates, max_seg=self.max_seg,
                       cand=self._candidates(instance))

    def describe(self) -> dict:
        info = super().describe()
        info.update(n_destroy=self.n_destroy, iter_max=self.iter_max,
                    time_limit=self.time_limit, n_candidates=self.n_candidates,
                    max_seg=self.max_seg)
        return info
