"""Repo-level solver contract: every registered solver scores every instance.

Deep, solver-specific tests live next to the solver:
`tests/llm/EoH/` for `solvers/llm/EoH`.
"""

import numpy as np
import pytest

from solvers.llm.EoH.atsp.baselines import BASELINES, load_baseline
from solvers.llm.EoH.atsp.data import ATSPInstance
from solvers.llm.EoH.atsp.data.synthetic import generate_instance, reference_cost
from solvers.llm.EoH.atsp.registry import TASK_SUMMARY, TASKS, build_problem

FAST_PARAMS = {
    "construct": {"n_candidates": 8},
    "gls": {"time_limit": 0.4, "ite_max": 4, "n_candidates": 6},
    "aco": {"n_ants": 4, "iter_max": 2},
    "rnr": {"iter_max": 4, "time_limit": 0.4, "n_candidates": 6},
}


@pytest.fixture(scope="module")
def instance():
    dist = generate_instance("asymmetric_clustered", 20, seed=1)
    return ATSPInstance(name="clustered20", dist=dist,
                        ref_cost=reference_cost(dist, effort="low", seed=1),
                        ref_kind="heuristic", source="synthetic")


def test_registry_is_complete():
    assert set(TASKS) == set(TASK_SUMMARY) == set(BASELINES)
    assert set(TASKS) == {"construct", "gls", "aco", "rnr"}


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_solver_returns_a_finite_tour_cost(task, instance):
    problem = build_problem(task, [instance], FAST_PARAMS[task], timeout=60)
    cost = problem.solve_instance(load_baseline(task), instance)
    assert np.isfinite(cost) and cost > 0


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_solver_scores_a_gap(task, instance):
    problem = build_problem(task, [instance], FAST_PARAMS[task], timeout=60)
    score = problem.score(load_baseline(task), instance)
    assert np.isfinite(score)
    assert score == pytest.approx(
        instance.gap(problem.solve_instance(load_baseline(task), instance)), abs=50)


def test_search_solvers_beat_the_greedy_construction(instance):
    """GLS and ruin-and-recreate must improve on nearest neighbour."""
    greedy = build_problem("construct", [instance], FAST_PARAMS["construct"])
    greedy_cost = greedy.solve_instance(load_baseline("construct"), instance)

    for task in ("gls", "rnr"):
        problem = build_problem(task, [instance], FAST_PARAMS[task])
        assert problem.solve_instance(load_baseline(task), instance) < greedy_cost
