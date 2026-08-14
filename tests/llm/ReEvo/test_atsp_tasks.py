"""Tests for the ReEvo ATSP layer: prompts, configs, engines, evaluation.

The asymmetry tests matter most here: ReEvo's upstream engines were written for
symmetric TSP (2-opt reversal, pheromone deposited on both directions), so these
are the checks that would catch a regression back to the symmetric assumptions.
"""

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REEVO = os.path.join(ROOT, "solvers", "llm", "ReEvo")
if "atsp_utils" in sys.modules:
    del sys.modules["atsp_utils"]
for _p in (os.path.join(REEVO, "problems", "atsp_aco"),
           os.path.join(REEVO, "problems", "atsp_gls"),
           REEVO, ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from evaluation.llm.ReEvo.benchmark_runner import (  # noqa: E402
    TASKS,
    load_callable,
    run_benchmark,
    seed_heuristic,
)
from atsp.data import ATSPInstance  # noqa: E402
from atsp.data.synthetic import generate_instance, reference_cost  # noqa: E402
from atsp.engines import is_valid_tour, tour_cost  # noqa: E402

WHITE_BOX = ["atsp_gls", "atsp_aco", "atsp_constructive"]
ALL_TASKS = WHITE_BOX + ["atsp_aco_black_box"]


@pytest.fixture(scope="module")
def instance():
    dist = generate_instance("asymmetric_clustered", 25, seed=11)
    return ATSPInstance(name="c25", dist=dist,
                        ref_cost=reference_cost(dist, effort="low", seed=11),
                        ref_kind="heuristic", source="synthetic")


# ── prompts and configs ───────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ALL_TASKS)
def test_prompt_files_exist(task):
    for name in ("seed_func.txt", "func_signature.txt", "func_desc.txt"):
        assert os.path.exists(os.path.join(REEVO, "prompts", task, name)), name


@pytest.mark.parametrize("task", WHITE_BOX)
def test_white_box_prompts_state_the_asymmetry(task):
    desc = open(os.path.join(REEVO, "prompts", task, "func_desc.txt"),
                encoding="utf-8").read()
    assert "NOT equal" in desc
    assert "distance_matrix[i][j]" in desc


def test_black_box_prompt_reveals_nothing_about_routing():
    desc = open(os.path.join(REEVO, "prompts", "atsp_aco_black_box",
                             "func_desc.txt"), encoding="utf-8").read().lower()
    for leak in ("tsp", "tour", "city", "distance", "asymmetric"):
        assert leak not in desc, f"black-box description leaks {leak!r}"


def problem_cfg(task: str) -> dict:
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "llm", "ReEvo", "cfg",
                                            "problem", f"{task}.yaml"), encoding="utf-8"))


@pytest.mark.parametrize("task", ALL_TASKS)
def test_config_is_wired(task):
    cfg = problem_cfg(task)
    assert cfg["obj_type"] == "min"
    assert cfg["problem_name"].startswith("atsp")
    assert os.path.isdir(os.path.join(REEVO, "problems", cfg["problem_name"]))


@pytest.mark.parametrize("task", ALL_TASKS)
def test_every_task_sets_its_own_kill_deadline(task):
    """`timeout: ${problem.timeout}` in config.yaml — a missing one breaks a run.

    Upstream's single 20 s deadline suits numba engines on five small
    instances; here a GLS evaluation legitimately takes ~80 s, and a deadline
    below that fails every candidate, seed included.
    """
    assert problem_cfg(task)["timeout"] >= 60
    root_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "llm", "ReEvo", "cfg",
                                                "config.yaml"), encoding="utf-8"))
    assert root_cfg["timeout"] == "${problem.timeout}"


def test_the_two_atsp_copies_have_not_drifted():
    """EoH and ReEvo keep separate `atsp/` trees; the data layer must match.

    Nothing enforces it at runtime, and a divergence here is silent and
    poisonous: the two frameworks would train on different matrices while the
    comparison table claims they trained on the same ones. `engines/__init__.py`
    is exempt — ReEvo deliberately exports a subset.
    """
    import filecmp

    eoh = os.path.join(ROOT, "solvers", "llm", "EoH", "atsp", "data")
    reevo = os.path.join(REEVO, "atsp", "data")
    for name in sorted(os.listdir(eoh)):
        if not name.endswith(".py"):
            continue
        assert filecmp.cmp(os.path.join(eoh, name), os.path.join(reevo, name),
                           shallow=False), f"atsp/data/{name} has drifted"


def _get_reevo_atsp_utils():
    path = os.path.join(REEVO, "atsp_utils.py")
    spec = importlib.util.spec_from_file_location("reevo_atsp_utils_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("task", ALL_TASKS)
def test_training_uses_the_whole_split(task):
    """The configured `problem_size` must not quietly shrink the training set.

    ReEvo passes `problem_size` to eval.py, and upstream uses it to select one
    dataset of one size. Filtering here would train every heuristic on small
    instances only — the overfitting that cost the EoH side a third of the
    benchmark — so the ATSP configs set 0, meaning "the whole split".
    """
    atsp_utils = _get_reevo_atsp_utils()
    load_instances = atsp_utils.load_instances

    unfiltered = load_instances("train", 0)
    configured = load_instances("train", problem_cfg(task)["problem_size"])
    assert len(configured) == len(unfiltered) > 0
    assert max(i.n for i in configured) > 100, "training needs instances beyond n=100"


def test_held_out_instances_are_not_trained_on():
    """rbg358 and rbg443 carry the generalisation claim; keep them unseen.

    Training moved onto the real distribution because no synthetic surrogate
    ranked heuristics the way the target does. That is defensible only while
    the reported claim rests on instances the search never saw.
    """
    atsp_utils = _get_reevo_atsp_utils()
    load_instances = atsp_utils.load_instances

    seen = {i.name for i in load_instances("train")}
    assert "rbg358" not in seen and "rbg443" not in seen


def test_default_model_is_gpt_4o_mini():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "llm", "ReEvo", "cfg", "llm_client", "openai.yaml"),
                              encoding="utf-8"))
    assert cfg["model"] == "gpt-4o-mini"


@pytest.mark.parametrize("task", WHITE_BOX)
def test_seed_heuristic_compiles(task):
    func = load_callable(seed_heuristic(task), TASKS[task]["entry"]
                         if task in TASKS else "heuristics")
    assert callable(func)


# ── asymmetric safety of the ported engines ───────────────────────────────────

def test_gls_returns_a_valid_tour_and_true_cost(instance):
    from gls import guided_local_search
    tour, cost = guided_local_search(instance.dist, instance.dist.copy(),
                                     perturbation_moves=5, iter_limit=10,
                                     time_limit=5.0)
    assert is_valid_tour(tour, instance.n)
    assert cost == pytest.approx(tour_cost(tour, instance.dist))


def test_gls_rejects_a_malformed_guide(instance):
    from gls import guided_local_search
    with pytest.raises(ValueError):
        guided_local_search(instance.dist, np.zeros((3, 3)), iter_limit=1)
    with pytest.raises(ValueError):
        guided_local_search(instance.dist, instance.dist * np.nan, iter_limit=1)


def test_aco_pheromone_stays_directed(instance):
    from aco import ACO
    dist = instance.dist.copy()
    np.fill_diagonal(dist, 1.0)
    aco = ACO(instance.dist, 1.0 / dist, n_ants=4, seed=0)
    aco.run(3)
    assert not np.allclose(aco.pheromone, aco.pheromone.T), \
        "pheromone was deposited on both directions"


def test_aco_tolerates_an_infinite_diagonal(instance):
    """`1 / distance_matrix` is the seed; the zero diagonal must not break it."""
    from aco import ACO
    heu = np.divide(1.0, instance.dist, out=np.full_like(instance.dist, np.inf),
                    where=instance.dist != 0)
    aco = ACO(instance.dist, heu, n_ants=3, seed=0)
    assert np.isfinite(aco.run(2))


def test_aco_rejects_a_wrongly_shaped_heuristic(instance):
    from aco import ACO
    with pytest.raises(ValueError):
        ACO(instance.dist, np.ones((3, 3)), n_ants=3)


# ── evaluation path ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["atsp_gls", "atsp_aco", "atsp_constructive"])
def test_seed_heuristic_scores_through_the_benchmark_runner(task, instance):
    fast = {"atsp_gls": {"perturbation_moves": 3, "iter_limit": 3, "time_limit": 2.0},
            "atsp_aco": {"n_iterations": 2, "n_ants": 4, "time_limit": 5.0},
            "atsp_constructive": {}}[task]
    records = run_benchmark(task, seed_heuristic(task), [instance], fast,
                            log=lambda *_: None)
    assert records[0]["status"] == "ok"
    assert np.isfinite(records[0]["cost"])


def test_a_broken_heuristic_is_recorded_not_raised(instance):
    broken = "import numpy as np\n\ndef heuristics(d):\n    raise RuntimeError('nope')\n"
    records = run_benchmark("atsp_gls", broken, [instance],
                            {"perturbation_moves": 1, "iter_limit": 1,
                             "time_limit": 1.0}, log=lambda *_: None)
    assert records[0]["status"] == "failed"
    assert "RuntimeError" in records[0]["error"]


@pytest.mark.parametrize("task", ["atsp_gls", "atsp_aco"])
def test_training_budget_is_not_wall_clock(task):
    """A training score must depend on the heuristic, not on CPU contention.

    ReEvo evaluates a generation as concurrent subprocesses. With a wall-clock
    budget the seed — evaluated alone at iteration 0 — scored -2.74 while the
    identical code re-sampled later alongside nine rivals scored -1.23, a swing
    bigger than the spread between candidates. The elitist could then never be
    displaced. Benchmarking still uses the clock: it runs on its own.
    """
    import importlib.util

    path = os.path.join(REEVO, "problems", task, "eval.py")
    spec = importlib.util.spec_from_file_location(f"{task}_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if task == "atsp_gls":
        atsp_utils = _get_reevo_atsp_utils()
        BENCHMARK_SECONDS = atsp_utils.BENCHMARK_SECONDS
        TRAIN_SECONDS = atsp_utils.TRAIN_SECONDS
        train_iter_limit = atsp_utils.train_iter_limit

        # the iteration budget must track instance size, since a fixed count is
        # a different effective budget at n=50 and at n=443
        assert train_iter_limit(50) > train_iter_limit(450)
        assert TRAIN_SECONDS <= BENCHMARK_SECONDS
        assert module.VAL_TIME_LIMIT > 0        # reporting budget is unchanged
    else:
        assert module.TRAIN_TIME_LIMIT is None
        assert module.N_ITERATIONS > 0


def test_the_same_guide_always_scores_the_same(instance):
    """The training engine must be deterministic run to run."""
    from gls import guided_local_search

    runs = [guided_local_search(instance.dist, instance.dist.copy(),
                                perturbation_moves=30, iter_limit=20,
                                time_limit=None)[1] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_reevo_runs_are_not_filed_under_eoh_in_the_merged_table(tmp_path):
    """One table over runs/llm must separate the frameworks, not merge them.

    Both write the same `eval_test.json` schema; the `framework` field is what
    keeps ReEvo's rows out of EoH's bucket when `--runs-root runs/llm` picks up
    `<framework>/<task>/<run>/`.
    """
    import json

    from evaluation.llm.EoH.stats_analysis import aggregate_by_task, collect

    for framework in ("EoH", "ReEvo"):
        run_dir = tmp_path / framework / "atsp_gls" / "20260101-000000"
        run_dir.mkdir(parents=True)
        (run_dir / "eval_test.json").write_text(json.dumps({
            "task": "atsp_gls", "framework": framework,
            "heuristic": "best_heuristic.py", "run_dir": str(run_dir),
            "summary": {"mean_gap_percent": 1.0, "median_gap_percent": 1.0,
                        "n_solved": 19, "n_instances": 19, "n_optimal": 0,
                        "total_seconds": 1.0},
            "records": []}))

    aggregated = aggregate_by_task(collect(str(tmp_path)))
    assert set(aggregated["atsp_gls"]) == {"EoH", "ReEvo"}
    assert aggregated["atsp_gls"]["ReEvo"]["n_runs"] == 1


@pytest.mark.parametrize("task", WHITE_BOX)
def test_eval_script_prints_the_objective_last(task, tmp_path):
    """ReEvo parses float(stdout.split('\\n')[-2]); the contract must hold.

    Passes `problem_size=50` rather than the configured 0 purely to keep the
    suite quick — the n=50 subset exercises the same code path, and the stdout
    contract does not depend on which instances were solved. Coverage of the
    real (whole-split) setting is `test_training_uses_the_whole_mixed_split`.
    """
    gpt_path = os.path.join(REEVO, "problems", task, "gpt.py")
    backup = open(gpt_path, encoding="utf-8").read() if os.path.exists(gpt_path) else None
    try:
        with open(gpt_path, "w", encoding="utf-8") as fh:
            fh.write(seed_heuristic(task))
        out = subprocess.run(
            [sys.executable, "-u", os.path.join(REEVO, "problems", task, "eval.py"),
             "50", REEVO, "train"],
            capture_output=True, text=True, timeout=600, cwd=REEVO)
        assert out.returncode == 0, out.stderr[-2000:]
        float(out.stdout.split("\n")[-2])          # the ReEvo contract
    finally:
        if backup is not None:
            with open(gpt_path, "w", encoding="utf-8") as fh:
                fh.write(backup)
