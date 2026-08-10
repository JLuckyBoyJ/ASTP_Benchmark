"""Tests for the four ATSP tasks: prompts, templates, baselines and fitness."""

import numpy as np
import pytest

from solvers.llm.EoH.atsp.baselines import BASELINES, baseline_code, load_baseline
from solvers.llm.EoH.atsp.data import ATSPInstance
from solvers.llm.EoH.atsp.data.synthetic import generate_instance, reference_cost
from solvers.llm.EoH.atsp.problems import ATSP_CONTEXT
from solvers.llm.EoH.atsp.registry import TASK_SUMMARY, TASKS, build_problem

#: budgets small enough for a fast test run
FAST_PARAMS = {
    "construct": {"n_candidates": 10},
    "gls": {"time_limit": 0.5, "ite_max": 5, "n_candidates": 6},
    "aco": {"n_ants": 5, "iter_max": 3},
    "rnr": {"iter_max": 5, "time_limit": 0.5, "n_candidates": 6},
}


@pytest.fixture(scope="module")
def instances():
    out = []
    for idx in range(2):
        dist = generate_instance("uniform", 25, seed=300 + idx)
        out.append(ATSPInstance(name=f"t{idx}", dist=dist,
                                ref_cost=reference_cost(dist, effort="low",
                                                        seed=300 + idx),
                                ref_kind="heuristic", source="synthetic"))
    return out


# ── prompts ───────────────────────────────────────────────────────────────────

def test_shared_context_states_the_asymmetry():
    assert "distance_matrix[i][j]" in ATSP_CONTEXT
    assert "NOT equal" in ATSP_CONTEXT


@pytest.mark.parametrize("task", sorted(TASKS))
def test_task_description_is_atsp_specific(task):
    description = TASKS[task].task_description
    assert description.startswith(ATSP_CONTEXT)
    assert "asymmetr" in description.lower()
    assert "TSP" in description
    # the task-specific half must actually say something beyond the context
    assert len(description) > len(ATSP_CONTEXT) + 200


@pytest.mark.parametrize("task", sorted(TASKS))
def test_template_compiles_and_defines_the_entry_point(task):
    cls = TASKS[task]
    namespace: dict = {}
    exec(compile(cls.template_program, "<template>", "exec"), namespace)
    assert callable(namespace[cls.entry_point])


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_has_a_summary(task):
    assert TASK_SUMMARY[task].endswith(".")


def test_gls_prompt_warns_against_symmetrising_penalties():
    text = TASKS["gls"].task_description
    assert "separate penalty counters" in text or "different arcs" in text


def test_aco_prompt_forbids_symmetrising_pheromone():
    assert "never symmetrise" in TASKS["aco"].task_description


# ── baselines and fitness ─────────────────────────────────────────────────────

@pytest.mark.parametrize("task", sorted(TASKS))
def test_baseline_entry_point_matches_the_task(task):
    assert BASELINES[task]["entry_point"] == TASKS[task].entry_point


@pytest.mark.parametrize("task", sorted(TASKS))
def test_baseline_produces_a_finite_positive_gap(task, instances):
    problem = build_problem(task, instances, FAST_PARAMS[task], timeout=60)
    fitness = problem.evaluate_program("", load_baseline(task))
    assert fitness is not None and np.isfinite(fitness)
    assert problem.fitness_name == "mean_gap_percent"


@pytest.mark.parametrize("task", sorted(TASKS))
def test_evaluate_compiles_source_like_the_framework_does(task, instances):
    """EoH calls evaluate(code_string); it must reach the same number."""
    problem = build_problem(task, instances, FAST_PARAMS[task], timeout=60)
    assert problem.evaluate(baseline_code(task)) is not None


@pytest.mark.parametrize("task", sorted(TASKS))
def test_describe_reports_the_settings(task, instances):
    problem = build_problem(task, instances, FAST_PARAMS[task], timeout=42)
    info = problem.describe()
    assert info["task"] == task
    assert info["n_train_instances"] == 2
    assert info["timeout_s"] == 42
    assert info["fitness"] == "mean_gap_percent"


def test_fitness_falls_back_to_raw_cost_without_a_reference():
    dist = generate_instance("uniform", 15, seed=77)
    unreferenced = [ATSPInstance(name="x", dist=dist, ref_cost=None,
                                 ref_kind="unknown", source="synthetic")]
    problem = build_problem("construct", unreferenced, {"n_candidates": 5})
    assert problem.fitness_name == "mean_tour_cost"
    assert problem.evaluate_program("", load_baseline("construct")) > 0


def test_a_broken_heuristic_evaluates_to_none(instances):
    problem = build_problem("construct", instances, {"n_candidates": 5}, timeout=10)
    assert problem.evaluate("def select_next_node(*a):\n    raise RuntimeError()\n") is None
    assert problem.evaluate("this is not python") is None
    assert problem.evaluate("def other_name(x):\n    return x\n") is None


def test_empty_instance_list_is_rejected():
    with pytest.raises(ValueError):
        build_problem("construct", [], {})


def test_unknown_task_is_rejected(instances):
    with pytest.raises(ValueError):
        build_problem("tsp", instances, {})
