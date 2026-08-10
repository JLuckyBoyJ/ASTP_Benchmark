"""Tests for EoH run configuration: defaults, `extends`, env vars, overrides."""

import os

import pytest
import yaml

from solvers.llm.EoH.atsp.config import (
    DEFAULTS,
    apply_override,
    deep_merge,
    dump_config,
    load_config,
    load_dotenv,
    parse_dotenv,
    repo_root,
)
from solvers.llm.EoH.atsp.registry import TASKS

ROOT = repo_root()
CONFIG_DIR = os.path.join(ROOT, "configs", "llm", "EoH")


def config_path(task: str) -> str:
    return os.path.join(CONFIG_DIR, f"atsp_{task}.yaml")


@pytest.mark.parametrize("task", sorted(TASKS))
def test_shipped_config_matches_the_paper_defaults(task):
    config = load_config(config_path(task))
    assert config["task"]["name"] == task
    assert config["llm"]["model"] == "gpt-4o-mini"
    assert config["llm"]["api_endpoint"] == "api.openai.com"
    # EoH paper, TSP setting: 20 generations, population 10, 5 parents
    assert config["eoh"]["n_pop"] == 20
    assert config["eoh"]["pop_size"] == 10
    assert config["eoh"]["n_parents"] == 5
    assert config["eoh"]["operators"] == ["e1", "e2", "m1", "m2"]


@pytest.mark.parametrize("task", sorted(TASKS))
def test_train_is_synthetic_and_test_is_tsplib(task):
    """No benchmark instance may leak into the evolution loop."""
    config = load_config(config_path(task))
    assert config["data"]["train"]["source"] == "synthetic"
    assert config["data"]["test"]["source"] == "tsplib"
    assert config["data"]["test"]["dir"] == "data/raw/atsp"


@pytest.mark.parametrize("task", sorted(TASKS))
def test_eval_budget_is_at_least_the_training_budget(task):
    config = load_config(config_path(task))
    train = config["task"]["params"]
    evaluate = config["eval"]["params"]
    for key in ("time_limit", "ite_max", "iter_max", "n_runs"):
        if key in train and key in evaluate and train[key] and evaluate[key]:
            assert evaluate[key] >= train[key], f"{task}: eval {key} < train {key}"


@pytest.mark.parametrize("task", sorted(TASKS))
def test_run_output_root_is_namespaced(task):
    assert load_config(config_path(task))["run"]["output_root"] == "runs/llm/EoH"


def test_extends_chain_pulls_in_the_base_file():
    raw = yaml.safe_load(open(config_path("gls"), encoding="utf-8"))
    assert raw["extends"] == "base.yaml"
    assert "llm" not in raw                    # only base.yaml defines it
    assert load_config(config_path("gls"))["llm"]["model"] == "gpt-4o-mini"


def test_overrides_are_yaml_typed():
    config = load_config(config_path("gls"),
                         ["eoh.n_pop=3", "eoh.debug=true",
                          "task.params.time_limit=0.5",
                          "eoh.operators=[e1, m1]"])
    assert config["eoh"]["n_pop"] == 3
    assert config["eoh"]["debug"] is True
    assert config["task"]["params"]["time_limit"] == 0.5
    assert config["eoh"]["operators"] == ["e1", "m1"]


def test_override_can_create_a_missing_key():
    config = load_config(config_path("aco"), ["run.tag=pilot"])
    assert config["run"]["tag"] == "pilot"


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError):
        apply_override({}, "no-equals-sign")


def test_missing_env_var_becomes_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_config(config_path("aco"), use_dotenv=False)["llm"]["api_key"] is None


def test_env_var_is_substituted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    assert load_config(config_path("aco"))["llm"]["api_key"] == "sk-test-123"


# ── .env handling ─────────────────────────────────────────────────────────────

def test_dotenv_parser_handles_the_usual_shapes():
    parsed = parse_dotenv(
        "# a comment\n"
        "\n"
        "OPENAI_API_KEY=sk-plain\n"
        "export EXPORTED=yes\n"
        'QUOTED="with spaces"\n'
        "SINGLE='single'\n"
        "EMPTY=\n"
        "not a variable line\n"
        "1BAD=nope\n"
    )
    assert parsed == {"OPENAI_API_KEY": "sk-plain", "EXPORTED": "yes",
                      "QUOTED": "with spaces", "SINGLE": "single", "EMPTY": ""}


def test_dotenv_does_not_expand_dollar_signs():
    assert parse_dotenv("KEY=sk-$LITERAL$")["KEY"] == "sk-$LITERAL$"


def test_dotenv_populates_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n")
    assert load_dotenv(path=str(env_file)) == [str(env_file)]
    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n")
    load_dotenv(path=str(env_file))
    assert os.environ["OPENAI_API_KEY"] == "sk-from-shell"


def test_env_file_variable_selects_the_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / "other.env"
    env_file.write_text("OPENAI_API_KEY=sk-other\n")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    config = load_config(config_path("gls"))
    assert config["llm"]["api_key"] == "sk-other"
    assert config["_dotenv"] == [str(env_file)]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_missing_dotenv_is_not_an_error(monkeypatch):
    monkeypatch.setenv("ENV_FILE", "/nonexistent/.env")
    assert load_config(config_path("gls"))["_dotenv"] == []


def test_dotenv_example_is_tracked_and_has_no_real_key():
    example = os.path.join(ROOT, "envs", ".env.example")
    assert os.path.exists(example), "envs/.env.example must be committed"
    parsed = parse_dotenv(open(example, encoding="utf-8").read())
    assert "OPENAI_API_KEY" in parsed
    assert parsed["OPENAI_API_KEY"] == "sk-replace-me", "a real key leaked into the template"


def test_dump_redacts_the_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    text = dump_config(load_config(config_path("rnr")))
    assert "sk-super-secret" not in text
    assert "redacted" in text


def test_deep_merge_keeps_untouched_branches():
    merged = deep_merge(DEFAULTS, {"eoh": {"n_pop": 1}})
    assert merged["eoh"]["n_pop"] == 1
    assert merged["eoh"]["pop_size"] == DEFAULTS["eoh"]["pop_size"]
    assert DEFAULTS["eoh"]["n_pop"] == 20, "DEFAULTS must not be mutated"
