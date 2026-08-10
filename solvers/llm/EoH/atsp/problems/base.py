"""Shared plumbing for the four ATSP tasks.

Every task is an ordinary EoH ``BaseProblem``: it declares the code template the
LLM must fill in, a natural-language task description (the *thought* prompt),
and an evaluation routine returning a scalar fitness where **lower is better**.

Fitness = mean optimality gap (%) over the training instances, matching the
EoH paper's TSP setting. If any training instance lacks a reference cost the
fitness falls back to the mean raw tour cost (and says so via ``fitness_name``).
"""

from __future__ import annotations

import numpy as np

from .. import _bootstrap  # noqa: F401  (puts the vendored `eoh` package on sys.path)
from eoh import BaseProblem

#: Prepended to every task description so the LLM always knows the setting.
ATSP_CONTEXT = (
    "You are designing a heuristic for the Asymmetric Travelling Salesman Problem (ATSP). "
    "An ATSP instance is a directed cost matrix where distance_matrix[i][j] is the cost of "
    "travelling FROM city i TO city j, and distance_matrix[i][j] is generally NOT equal to "
    "distance_matrix[j][i]. The objective is to find a closed tour that visits every city "
    "exactly once and returns to the start with minimum total cost. "
    "Because the matrix is asymmetric, heuristics that implicitly assume symmetry are "
    "misleading: an arc is not an undirected edge, and reversing part of a tour changes its "
    "cost. A strong heuristic should exploit the difference between the two directions, "
    "for example by comparing outgoing costs with incoming costs."
)


class ATSPProblem(BaseProblem):
    """Base class for the ATSP EoH tasks."""

    #: short identifier used in configs, run directories and result tables
    task_name: str = "atsp"
    #: name of the function the LLM must produce (entry point of the template)
    entry_point: str = "heuristic"

    def __init__(self, instances, timeout: int = 60, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        if not instances:
            raise ValueError("ATSP problem created with an empty instance list")
        self.instances = list(instances)
        self._use_gap = all(ins.ref_cost for ins in self.instances)

    # ── interface ────────────────────────────────────────────────────────────

    @property
    def fitness_name(self) -> str:
        return "mean_gap_percent" if self._use_gap else "mean_tour_cost"

    def solve_instance(self, callable_func, instance) -> float:
        """Run the task's algorithm with ``callable_func`` on one instance."""
        raise NotImplementedError

    def score(self, callable_func, instance) -> float:
        """Instance score: optimality gap (%) when a reference cost is known."""
        cost = float(self.solve_instance(callable_func, instance))
        if not np.isfinite(cost):
            raise ValueError("non-finite tour cost")
        return instance.gap(cost) if self._use_gap else cost

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        scores = [self.score(callable_func, ins) for ins in self.instances]
        value = float(np.mean(scores))
        return value if np.isfinite(value) else None

    # ── helpers ──────────────────────────────────────────────────────────────

    def describe(self) -> dict:
        return {
            "task": self.task_name,
            "entry_point": self.entry_point,
            "fitness": self.fitness_name,
            "n_train_instances": len(self.instances),
            "instance_sizes": sorted({ins.n for ins in self.instances}),
            "reference": sorted({ins.ref_kind for ins in self.instances}),
            "timeout_s": self.timeout,
        }
