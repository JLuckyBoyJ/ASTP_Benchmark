"""Tests for MCTS-AHD configuration files, Hydra defaults, evaluation budgets, and solver registry integration."""

import os
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AHD = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")
TASKS = ("atsp_constructive", "atsp_gls", "atsp_kgls", "atsp_aco")
IMPROVEMENT_TASKS = ("atsp_gls", "atsp_kgls")


@pytest.mark.parametrize("name", ["tsplib", "synthetic", "mcts_ahd_native"])
def test_every_data_config_declares_three_splits(name):
    import sys
    if AHD not in sys.path:
        sys.path.insert(0, AHD)
    import atsp_utils
    config = atsp_utils.load_config("data", name)
    for mood in ("train", "val", "test"):
        assert config.get(mood), f"{name}.yaml has no {mood} split"


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_an_evaluation_budget(task):
    import sys
    if AHD not in sys.path:
        sys.path.insert(0, AHD)
    import atsp_utils
    for mood in ("train", "val", "test"):
        settings = atsp_utils.budget(task, mood)
        assert isinstance(settings, dict)


@pytest.mark.parametrize("task", IMPROVEMENT_TASKS)
def test_training_budget_tracks_the_benchmark(task):
    import sys
    if AHD not in sys.path:
        sys.path.insert(0, AHD)
    import atsp_utils
    settings = atsp_utils.budget(task, "train")
    assert settings["iter_limit"] == "auto"
    small = atsp_utils.resolve_iter_limit("auto", 50)
    large = atsp_utils.resolve_iter_limit("auto", 400)
    assert small > large > 0, "a bigger instance must get fewer iterations"
    assert atsp_utils.resolve_iter_limit(1000, 50) == 1000


def test_gls_and_kgls_get_the_same_engine_budget():
    import sys
    if AHD not in sys.path:
        sys.path.insert(0, AHD)
    import atsp_utils
    for mood in ("train", "val", "test"):
        gls = atsp_utils.budget("atsp_gls", mood)
        kgls = atsp_utils.budget("atsp_kgls", mood)
        assert gls == kgls, (
            f"atsp_gls and atsp_kgls have different {mood} budgets: {gls} vs {kgls}"
        )


def test_data_config_selection_honours_the_environment(monkeypatch):
    import sys
    if AHD not in sys.path:
        sys.path.insert(0, AHD)
    import atsp_utils
    monkeypatch.setenv("ATSP_DATA", "synthetic")
    assert atsp_utils.data_config_name() == "synthetic"
    monkeypatch.delenv("ATSP_DATA")
    monkeypatch.setenv("ATSP_TRAIN_SPLIT", "synth")
    assert atsp_utils.data_config_name() == "synthetic"
    monkeypatch.delenv("ATSP_TRAIN_SPLIT")
    assert atsp_utils.data_config_name() == "tsplib"


def test_solver_registry_lists_every_task():
    path = os.path.join(ROOT, "configs", "solvers", "llm.yaml")
    registry = yaml.safe_load(open(path, encoding="utf-8"))["mcts_ahd"]
    assert set(registry["configs"]) == set(TASKS)
    assert set(registry["evaluation_budgets"]) == set(TASKS)
    for rel in list(registry["configs"].values()) + \
               list(registry["evaluation_budgets"].values()) + \
               list(registry["data_configs"].values()):
        assert os.path.isfile(os.path.join(ROOT, rel)), rel
    for key in ("entry_point", "run_script", "eval_script", "data_script"):
        assert os.path.isfile(os.path.join(ROOT, registry[key])), registry[key]


def test_config_exposes_the_papers_defaults():
    path = os.path.join(ROOT, "configs", "llm", "MCTS-AHD", "cfg", "config.yaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    assert cfg["init_pop_size"] == 4
    assert cfg["pop_size"] == 10
    assert cfg["exploration_constant"] == 0.1
    assert cfg["progressive_widening_alpha"] == 0.5
    assert cfg["expansion_children"] == 2
    assert cfg["data"] in ("synthetic", "tsplib")


@pytest.mark.parametrize("task", TASKS)
def test_problem_config_matches_the_task(task):
    path = os.path.join(ROOT, "configs", "llm", "MCTS-AHD", "cfg", "problem",
                        f"{task}.yaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    assert cfg["problem_name"] == task
    assert cfg["obj_type"] == "min"
    assert cfg["timeout"] > 0
    signature = open(os.path.join(AHD, "prompts", task, "func_signature.txt"),
                     encoding="utf-8").read()
    assert cfg["func_name"] in signature
    assert "Asymmetric" in cfg["description"]
