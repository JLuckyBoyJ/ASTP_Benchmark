"""Task registry: ``task name -> EoH problem class``."""

from __future__ import annotations

from .problems import (
    ATSPAntColony,
    ATSPConstruct,
    ATSPGuidedLocalSearch,
    ATSPProblem,
    ATSPRuinAndRecreate,
)

TASKS: dict[str, type[ATSPProblem]] = {
    "construct": ATSPConstruct,
    "gls": ATSPGuidedLocalSearch,
    "aco": ATSPAntColony,
    "rnr": ATSPRuinAndRecreate,
}

TASK_SUMMARY = {
    "construct": "Design the next-city rule of a greedy ATSP tour construction.",
    "gls": "Design the cost-matrix update rule of an ATSP guided local search.",
    "aco": "Design the directed pheromone update rule of ACO for ATSP.",
    "rnr": "Design the destroy operator of an ATSP ruin-and-recreate search.",
}


def get_problem_class(task: str) -> type[ATSPProblem]:
    try:
        return TASKS[task]
    except KeyError:
        raise ValueError(
            f"Unknown ATSP task {task!r}. Available: {', '.join(sorted(TASKS))}"
        ) from None


def build_problem(task: str, instances, params: dict | None = None,
                  timeout: int | None = None, n_processes: int = 1) -> ATSPProblem:
    """Instantiate a task with the ``task.params`` block of a config."""
    cls = get_problem_class(task)
    kwargs = dict(params or {})
    if timeout is not None:
        kwargs["timeout"] = int(timeout)
    kwargs["n_processes"] = int(n_processes)
    return cls(instances, **kwargs)
