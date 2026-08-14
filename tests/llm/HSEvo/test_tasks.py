"""Tests for HSEvo ATSP task prompts, seed heuristic compilation, and evaluation benchmark runner."""

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HSEVO = os.path.join(ROOT, "solvers", "llm", "HSEvo")

for _p in (HSEVO, os.path.join(HSEVO, "problems", "atsp_gls"),
           os.path.join(HSEVO, "problems", "atsp_aco"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.llm.HSEvo.benchmark_runner import TASKS, load_callable, run_benchmark, seed_heuristic
from atsp.data.synthetic import generate_instance, reference_cost
from atsp.data import ATSPInstance

WHITE_BOX = ["atsp_gls", "atsp_aco", "atsp_constructive"]


@pytest.fixture(scope="module")
def instance():
    dist = generate_instance("asymmetric_clustered", 25, seed=11)
    return ATSPInstance(name="c25", dist=dist,
                        ref_cost=reference_cost(dist, effort="low", seed=11),
                        ref_kind="heuristic", source="synthetic")


@pytest.mark.parametrize("task", WHITE_BOX)
def test_prompt_files_exist(task):
    for name in ("seed_func.txt", "func_signature.txt", "func_desc.txt"):
        path = os.path.join(HSEVO, "prompts", task, name)
        assert os.path.exists(path), f"missing prompt file: {path}"


@pytest.mark.parametrize("task", WHITE_BOX)
def test_prompts_state_asymmetry(task):
    desc = open(os.path.join(HSEVO, "prompts", task, "func_desc.txt"), encoding="utf-8").read()
    assert "NOT equal" in desc or "Asymmetric" in desc


@pytest.mark.parametrize("task", WHITE_BOX)
def test_seed_heuristic_compiles(task):
    code = seed_heuristic(task)
    entry = TASKS[task]["entry"]
    func = load_callable(code, entry)
    assert callable(func)


@pytest.mark.parametrize("task", WHITE_BOX)
def test_seed_heuristic_scores_through_benchmark_runner(task, instance):
    code = seed_heuristic(task)
    records = run_benchmark(task, code, [instance])
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["cost"] is not None
    assert isinstance(records[0]["cost"], float)
