"""MoH configuration: Hydra defaults, problem YAMLs, budgets, registry."""

import os
import sys

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOH = os.path.join(ROOT, "solvers", "llm", "MoH")
CFG = os.path.join(ROOT, "configs", "llm", "MoH", "cfg")
TASKS = ("atsp_constructive", "atsp_gls", "atsp_kgls")
IMPROVEMENT_TASKS = ("atsp_gls", "atsp_kgls")

if MOH not in sys.path:
    sys.path.insert(0, MOH)


@pytest.mark.parametrize("name", ["synthetic", "tsplib", "multisize"])
def test_every_data_config_declares_three_splits(name):
    import atsp_utils
    config = atsp_utils.load_config("data", name)
    for mood in ("train", "val", "test"):
        assert config.get(mood), f"{name}.yaml has no {mood} split"


@pytest.mark.parametrize("name", ["synthetic", "tsplib", "multisize"])
def test_train_is_an_anchor_to_val(name):
    """MoH has one in-search split; the configs must not let the two drift."""
    import atsp_utils
    config = atsp_utils.load_config("data", name)
    assert config["train"] == config["val"], (
        f"{name}.yaml: train and val differ. MoH scores every utility call on "
        f"D_i (the val split, Eq. 1), so a separate train split would never be "
        f"used and prepare_data.py would generate instances nothing reads.")


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_an_evaluation_budget(task):
    import atsp_utils
    for mood in ("train", "val", "test"):
        assert isinstance(atsp_utils.budget(task, mood), dict)


@pytest.mark.parametrize("task", IMPROVEMENT_TASKS)
def test_search_budget_tracks_the_benchmark(task):
    import atsp_utils
    settings = atsp_utils.budget(task, "val")
    assert settings["iter_limit"] == "auto"
    small = atsp_utils.resolve_iter_limit("auto", 50)
    large = atsp_utils.resolve_iter_limit("auto", 400)
    assert small > large > 0, "a bigger instance must get fewer iterations"
    assert atsp_utils.resolve_iter_limit(1000, 50) == 1000


def test_gls_and_kgls_get_the_same_engine_budget():
    import atsp_utils
    for mood in ("train", "val", "test"):
        gls, kgls = atsp_utils.budget("atsp_gls", mood), atsp_utils.budget("atsp_kgls", mood)
        assert gls == kgls, (
            f"atsp_gls and atsp_kgls have different {mood} budgets: {gls} vs {kgls}")


@pytest.mark.parametrize("task", IMPROVEMENT_TASKS)
def test_engine_budget_matches_the_mcts_ahd_side(task):
    """The frameworks are only comparable if the engine budget is identical."""
    ours = yaml.safe_load(open(os.path.join(CFG, "evaluation", f"{task}.yaml"),
                               encoding="utf-8"))
    theirs_path = os.path.join(ROOT, "configs", "llm", "MCTS-AHD", "cfg",
                               "evaluation", f"{task}.yaml")
    if not os.path.isfile(theirs_path):
        pytest.skip("MCTS-AHD configs are not present")
    theirs = yaml.safe_load(open(theirs_path, encoding="utf-8"))
    assert ours["perturbation_moves"] == theirs["perturbation_moves"]
    assert ours["test"] == theirs["test"]
    assert ours["val"]["equivalent_seconds"] == theirs["train"]["equivalent_seconds"]


def test_data_config_selection_honours_the_environment(monkeypatch):
    import atsp_utils
    monkeypatch.setenv("ATSP_DATA", "tsplib")
    assert atsp_utils.data_config_name() == "tsplib"
    monkeypatch.delenv("ATSP_DATA")
    monkeypatch.setenv("ATSP_TRAIN_SPLIT", "synth")
    assert atsp_utils.data_config_name() == "synthetic"
    monkeypatch.delenv("ATSP_TRAIN_SPLIT")
    assert atsp_utils.data_config_name() == "synthetic"


def test_config_exposes_the_papers_defaults():
    cfg = yaml.safe_load(open(os.path.join(CFG, "config.yaml"), encoding="utf-8"))
    assert cfg["n_iterations"] == 10          # T = 10 (Section 4)
    assert cfg["pop_size"] == 10              # Section 3.3 / Table 5
    assert cfg["mode"] in ("train", "inference")
    assert cfg["max_optimizer_iterations"] == 10   # Appendix D.1


def test_data_is_a_config_group_not_a_bare_key():
    """`cfg/data/` is a directory under config_path, so Hydra treats `data` as a
    config group whether or not the defaults list says so. A plain `data:` key
    alongside it makes every `data=...` override fail with "No match in the
    defaults list", which is a confusing way to learn this."""
    cfg = yaml.safe_load(open(os.path.join(CFG, "config.yaml"), encoding="utf-8"))
    assert "data" not in cfg, "`data` must not be a top-level key as well as a group"
    selected = [d["data"] for d in cfg["defaults"]
                if isinstance(d, dict) and "data" in d]
    assert selected == ["synthetic"], f"expected `- data: synthetic`, got {selected}"


def test_the_smoke_configs_exist_and_are_cheap():
    """`--smoke` is only useful if it is actually fast."""
    data = yaml.safe_load(open(os.path.join(CFG, "data", "smoke.yaml"), encoding="utf-8"))
    budget = yaml.safe_load(open(os.path.join(CFG, "evaluation", "smoke.yaml"),
                                 encoding="utf-8"))
    assert {int(s["size"]) for s in data["val"]} == {20, 40}
    assert budget["val"]["iter_limit"] <= 50
    assert budget["val"]["time_limit"] <= 5


@pytest.mark.parametrize("task", TASKS)
def test_problem_config_matches_the_task(task):
    cfg = yaml.safe_load(open(os.path.join(CFG, "problem", f"{task}.yaml"),
                              encoding="utf-8"))
    assert cfg["problem_name"] == task
    assert cfg["obj_type"] == "min"
    assert cfg["timeout"] > 0
    assert "Asymmetric" in cfg["description"]

    # The subtasks are sizes, and every size needs a threshold to seed against.
    sizes = cfg["problem_size"]
    assert isinstance(sizes, list) and sizes, "problem_size must be a non-empty list"
    for size in sizes:
        assert size in cfg["threshold"], f"no seed threshold for size {size}"

    # The function name must be the one the prompt and the seed function use.
    desc = open(os.path.join(MOH, "prompts", task, "desc.txt"), encoding="utf-8").read()
    seed = open(os.path.join(MOH, "prompts", task, "seed_func.txt"), encoding="utf-8").read()
    assert cfg["func_name"] in desc
    assert f"def {cfg['func_name']}(" in seed


def test_default_model_is_the_benchmarks():
    cfg = yaml.safe_load(open(os.path.join(CFG, "llm_client", "openai.yaml"),
                              encoding="utf-8"))
    assert cfg["model"] == "gpt-4o-mini"


def test_runs_are_written_into_the_shared_tree():
    cfg = yaml.safe_load(open(os.path.join(CFG, "hydra", "output", "local.yaml"),
                              encoding="utf-8"))
    assert "runs/llm/MoH" in cfg["hydra"]["run"]["dir"]


def test_solver_registry_lists_every_task():
    path = os.path.join(ROOT, "configs", "solvers", "llm.yaml")
    registry = yaml.safe_load(open(path, encoding="utf-8"))["moh"]
    assert set(registry["configs"]) == set(TASKS)
    assert set(registry["evaluation_budgets"]) == set(TASKS)
    for rel in (list(registry["configs"].values())
                + list(registry["evaluation_budgets"].values())
                + list(registry["data_configs"].values())):
        assert os.path.isfile(os.path.join(ROOT, rel)), rel
    for key in ("entry_point", "run_script", "eval_script", "data_script"):
        assert os.path.isfile(os.path.join(ROOT, registry[key])), registry[key]
    assert registry["default_model"] == "gpt-4o-mini"
