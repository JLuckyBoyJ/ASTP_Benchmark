"""Tests for HSEvo configuration files, Hydra defaults, and solver registry integration."""

import os
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_ROOT = os.path.join(ROOT, "configs", "llm", "HSEvo", "cfg")
TASKS = ["atsp_gls", "atsp_constructive", "atsp_aco"]


def problem_cfg(task: str) -> dict:
    path = os.path.join(CONFIG_ROOT, "problem", f"{task}.yaml")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main_cfg() -> dict:
    path = os.path.join(CONFIG_ROOT, "config.yaml")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_main_config_defaults():
    cfg = main_cfg()
    assert cfg["algorithm"] == "hsevo"
    assert cfg["model"] == "gpt-4o-mini"
    assert cfg["pop_size"] == 10
    assert cfg["init_pop_size"] == 30
    assert cfg["mutation_rate"] == 0.5
    assert cfg["timeout"] == "${problem.timeout}"
    # Harmony Search parameters
    assert cfg["hm_size"] == 5
    assert cfg["hmcr"] == 0.7
    assert cfg["par"] == 0.5
    assert cfg["bandwidth"] == 0.2
    assert cfg["max_iter"] == 5


@pytest.mark.parametrize("task", TASKS)
def test_problem_configs_wired(task):
    cfg = problem_cfg(task)
    assert cfg["problem_name"] == task
    assert cfg["obj_type"] == "min"
    assert cfg["problem_size"] == 0
    assert cfg["timeout"] >= 60


def test_openai_llm_client_config():
    path = os.path.join(CONFIG_ROOT, "llm_client", "openai.yaml")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["model"] == "gpt-4o-mini"


def test_solver_registry_entry():
    path = os.path.join(ROOT, "configs", "solvers", "llm.yaml")
    with open(path, encoding="utf-8") as fh:
        reg = yaml.safe_load(fh)
    assert "hsevo" in reg
    hsevo_entry = reg["hsevo"]
    assert hsevo_entry["implementation"] == "solvers/llm/HSEvo"
    assert hsevo_entry["default_model"] == "gpt-4o-mini"
    assert "atsp_gls" in hsevo_entry["configs"]
    assert "hsevo" in reg["task_coverage"]["guided_local_search"]
    assert "hsevo" in reg["task_coverage"]["greedy_construction"]
    assert "hsevo" in reg["task_coverage"]["ant_colony_optimisation"]
