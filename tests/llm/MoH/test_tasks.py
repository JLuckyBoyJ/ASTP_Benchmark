"""MoH's downstream tasks: prompts, seed functions, evaluation contract."""

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOH = os.path.join(ROOT, "solvers", "llm", "MoH")
TASKS = ("atsp_constructive", "atsp_gls", "atsp_kgls")

for _p in (ROOT, MOH):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _require_synthetic():
    """Skip unless the default split's instances are already cached.

    Resolving a split *generates* what is missing, reference costs and all —
    minutes of work for a size the suite only wanted to inspect. Tests check the
    cache first so `pytest` stays fast and never silently starts a data build;
    `python python_scripts/llm/MoH/prepare_data.py` is where that belongs.
    """
    import pytest
    import atsp_utils
    from atsp.data.synthetic import dataset_filename

    specs = atsp_utils.val_split()
    specs = specs if isinstance(specs, (list, tuple)) else [specs]
    for spec in specs:
        if str((spec or {}).get("source", "synthetic")).lower() != "synthetic":
            continue
        path = os.path.join(ROOT, "data", "synthetic", spec.get("family", "uniform"),
                            dataset_filename(spec.get("family", "uniform"),
                                             int(spec.get("size", 50)),
                                             int(spec.get("count", 8)),
                                             int(spec.get("seed", 2024))))
        if not os.path.isfile(path):
            pytest.skip("synthetic instances are not built; run "
                        "python python_scripts/llm/MoH/prepare_data.py")

@pytest.mark.parametrize("task", TASKS)
def test_every_task_ships_the_four_prompt_files(task):
    for name in ("desc.txt", "task.txt", "plan.txt", "size.txt", "seed_func.txt"):
        path = os.path.join(MOH, "prompts", task, name)
        assert os.path.isfile(path), path
        assert os.path.getsize(path) > 0, path


@pytest.mark.parametrize("task", TASKS)
def test_the_task_prompt_forbids_randomness(task):
    """Both loops select on utility, so a heuristic whose score changes between
    evaluations makes the selection noise."""
    desc = open(os.path.join(MOH, "prompts", task, "desc.txt"), encoding="utf-8").read()
    assert "NEVER include any random" in desc
    assert "deterministic" in desc


@pytest.mark.parametrize("task", TASKS)
def test_the_task_prompt_states_the_matrix_is_directed(task):
    desc = open(os.path.join(MOH, "prompts", task, "desc.txt"), encoding="utf-8").read()
    assert "asymmetric" in desc.lower() or "directed" in desc.lower()
    assert "[i, j]" in desc or "[i][j]" in desc


@pytest.mark.parametrize("task", TASKS)
def test_the_seed_function_compiles_and_runs(task):
    from evaluation.llm.MoH import TASKS as RUNNER_TASKS, load_callable, seed_heuristic
    func = load_callable(seed_heuristic(task), RUNNER_TASKS[task]["entry"])

    rng = np.random.default_rng(0)
    n = 12
    dist = rng.integers(1, 100, size=(n, n)).astype(float)
    np.fill_diagonal(dist, 0.0)

    if task == "atsp_constructive":
        out = func(current_node=0, destination_node=0,
                   unvisited_nodes=set(range(1, n)), distance_matrix=dist)
        assert int(out) in range(1, n)
    elif task == "atsp_gls":
        out = np.asarray(func(dist.copy(), np.arange(n), np.zeros((n, n))))
        assert out.shape == (n, n) and np.all(np.isfinite(out))
    else:
        out = np.asarray(func(dist.copy(), np.arange(n), np.zeros((n, n)), 0))
        assert out.shape == (n,) and np.all(np.isfinite(out))


@pytest.mark.parametrize("task", TASKS)
def test_gpt_py_holds_a_runnable_seed(task):
    """`problems/<task>/gpt.py` is overwritten on every evaluation, but what is
    committed must run — it is what a bare `python eval.py` uses."""
    path = os.path.join(MOH, "problems", task, "gpt.py")
    source = open(path, encoding="utf-8").read()
    namespace = {}
    exec(compile(source, path, "exec"), namespace)
    entry = {"atsp_constructive": "select_next_node", "atsp_gls": "update_edge_distance",
             "atsp_kgls": "arc_badness"}[task]
    assert callable(namespace.get(entry))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.slow
def test_the_evaluation_contract_holds(task):
    """eval.py must print the objective as the last line of stdout — that is
    the whole interface between the search and the engines."""
    import atsp_utils
    _require_synthetic()
    sizes = atsp_utils.available_sizes("val")

    env = dict(os.environ, ATSP_DATA="synthetic", ATSP_EVAL_BUDGET=task,
               PYTHONPATH=os.pathsep.join([MOH, ROOT]))
    # Restore the committed seed before running, in case a previous test or run
    # left a generated heuristic in gpt.py.
    seed = open(os.path.join(MOH, "prompts", task, "seed_func.txt"),
                encoding="utf-8").read()
    with open(os.path.join(MOH, "problems", task, "gpt.py"), "w",
              encoding="utf-8") as fh:
        fh.write("import numpy as np\n\n" + seed)

    out = subprocess.run(
        [sys.executable, "-u", os.path.join(MOH, "problems", task, "eval.py"),
         str(min(sizes)), MOH, "val"],
        capture_output=True, text=True, cwd=MOH, env=env, timeout=900)
    assert out.returncode == 0, out.stdout + out.stderr
    last = out.stdout.strip().splitlines()[-1]
    value = float(last)                       # must parse; this is the contract
    assert np.isfinite(value)


def test_the_benchmark_runner_covers_every_task():
    from evaluation.llm.MoH import TASKS as RUNNER_TASKS, TASK_SUMMARY
    assert set(RUNNER_TASKS) == set(TASKS)
    assert set(TASK_SUMMARY) == set(TASKS)


def test_the_runner_reads_budgets_from_the_shared_config():
    """The runner and eval.py must agree, or a run's own validation number and
    the benchmark number would mean different things."""
    import atsp_utils
    from evaluation.llm.MoH import default_params
    assert default_params("atsp_gls", "test") == atsp_utils.budget("atsp_gls", "test")
