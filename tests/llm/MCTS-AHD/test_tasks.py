"""Tests for MCTS-AHD ATSP prompts, signatures, evaluation contract, and benchmark runner."""

import os
import re
import subprocess
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AHD = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")
TASKS = ("atsp_constructive", "atsp_gls", "atsp_kgls", "atsp_aco")

for _p in (AHD, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def instance():
    from atsp.data import load_tsplib_atsp
    found = load_tsplib_atsp(os.path.join(ROOT, "data", "raw", "atsp"),
                             best_known_path=os.path.join(ROOT, "data", "raw",
                                                          "atsp", "bestSolutions.txt"),
                             names=["ftv33"])
    assert found, "data/raw/atsp/ftv33.atsp is missing"
    return found[0]


@pytest.mark.parametrize("task", TASKS)
def test_prompt_files_exist(task):
    for name in ("func_signature.txt", "func_desc.txt", "seed_func.txt",
                 "external_knowledge.txt"):
        path = os.path.join(AHD, "prompts", task, name)
        assert os.path.isfile(path), f"missing {task}/{name}"
        assert os.path.getsize(path) > 0, f"empty {task}/{name}"


@pytest.mark.parametrize("task", TASKS)
def test_signature_parses_like_problem_adapter(task):
    raw = open(os.path.join(AHD, "prompts", task, "func_signature.txt"),
               encoding="utf-8").read().format(version=2).strip()
    match = re.match(r"^def +(.+?)\((.*)\) *-> *(.*?) *:", raw)
    assert match is not None, f"{task}: unparseable signature {raw!r}"
    inputs = [txt.split(":")[0].strip() for txt in match.group(2).split(",")]
    assert all(i.isidentifier() for i in inputs), inputs


@pytest.mark.parametrize("task", TASKS)
def test_seed_function_is_valid_python(task):
    code = open(os.path.join(AHD, "prompts", task, "seed_func.txt"),
                encoding="utf-8").read()
    namespace = {"np": np}
    exec(compile("import numpy as np\n" + code, "<seed>", "exec"), namespace)
    entry = [k for k in namespace if k.endswith("_v1")]
    assert entry, f"{task}: seed_func.txt defines no <name>_v1 function"


@pytest.mark.parametrize("task", TASKS)
def test_prompt_seed_matches_gpt_py(task):
    import ast

    prompt = open(os.path.join(AHD, "prompts", task, "seed_func.txt"),
                  encoding="utf-8").read()
    shipped = open(os.path.join(AHD, "problems", task, "gpt.py"),
                   encoding="utf-8").read()

    def body(text):
        tree = ast.parse(text)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assert len(funcs) == 1, f"{task}: expected exactly one function, got {len(funcs)}"
        statements = funcs[0].body
        if (statements and isinstance(statements[0], ast.Expr)
                and isinstance(statements[0].value, ast.Constant)
                and isinstance(statements[0].value.value, str)):
            statements = statements[1:]
        return [ast.dump(node) for node in statements]

    assert body(prompt) == body(shipped), (
        f"{task}: prompts/seed_func.txt and problems/gpt.py have drifted apart"
    )


@pytest.mark.parametrize("task", TASKS)
def test_eval_prints_a_parseable_objective_last(task):
    env = dict(os.environ)
    env["ATSP_TRAIN_SPLIT"] = "tsplib"
    script = os.path.join(AHD, "problems", task, "eval.py")
    proc = subprocess.run([sys.executable, "-u", script, "0", AHD, "val"],
                          capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    objective = float(proc.stdout.split("\n")[-2])
    assert objective == objective and objective >= 0


@pytest.mark.parametrize("task", TASKS)
def test_benchmark_runner_scores_the_seed(task, instance, tmp_path):
    from evaluation.llm.mcts_ahd import (
        run_benchmark, seed_heuristic, write_results, TASKS as BENCH_TASKS)

    assert task in BENCH_TASKS
    from evaluation.llm.mcts_ahd import default_params
    params = dict(default_params(task))
    if "iter_limit" in params:
        params.update(iter_limit=5, time_limit=5.0, perturbation_moves=5)
    if "n_iterations" in params:
        params.update(n_iterations=2, n_ants=4, time_limit=10.0)

    records = run_benchmark(task, seed_heuristic(task), [instance], params=params,
                            log=lambda *a: None)
    assert records[0]["status"] == "ok", records[0]["error"]
    assert records[0]["gap_percent"] >= 0

    written = write_results(str(tmp_path), "t", records, {"framework": "MCTS-AHD"})
    assert os.path.isfile(written["csv"]) and os.path.isfile(written["markdown"])


def test_a_broken_heuristic_is_recorded_not_raised(instance):
    from evaluation.llm.mcts_ahd import run_benchmark

    code = "def heuristics(distance_matrix):\n    raise RuntimeError('boom')\n"
    records = run_benchmark("atsp_gls", code, [instance],
                            params={"iter_limit": 2}, log=lambda *a: None)
    assert records[0]["status"] == "failed"
    assert "RuntimeError" in records[0]["error"]
