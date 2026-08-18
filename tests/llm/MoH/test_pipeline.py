"""MoH's machinery: populations, text helpers, the stub model, an offline run."""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOH = os.path.join(ROOT, "solvers", "llm", "MoH")

for _p in (ROOT, MOH):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── population management (Section 3.3) ──────────────────────────────────────

def test_population_keeps_the_best_and_caps_its_size():
    from utils.population import Pop
    pop = Pop(["t"], size=3)
    for utility in (5.0, 1.0, 9.0, 3.0, 7.0):
        pop.save_solution("t", f"idea {utility}", f"def f():\n    return {utility}",
                          utility)
    assert pop.get_subtask_size("t") == 3
    assert pop.get_best_solution("t")["utility"] == 1.0
    assert pop.best_utility("t") == 1.0


def test_population_rejects_duplicates_and_non_code():
    from utils.population import Pop
    pop = Pop(["t"], size=5)
    assert pop.save_solution("t", "a", "def f():\n    return 1", 1.0)
    assert not pop.save_solution("t", "a", "def f():\n    return 1", 1.0)
    assert not pop.save_solution("t", "a", "not code at all", 1.0)
    assert not pop.save_solution("t", "a", None, 1.0)


# ── the helpers the generated optimizers call by name ────────────────────────

def test_extract_code_and_idea_round_trip():
    from utils.utils import extract_code, extract_idea
    response = ("{A one-line idea.}\n\n```python\nimport numpy as np\n"
                "def f(x):\n    return x\n```")
    assert extract_idea(response) == "A one-line idea."
    assert "def f(x):" in extract_code(response)


def test_missing_numpy_import_is_repaired():
    """Models write np. without the import about a third of the time; upstream
    scored the resulting NameError as a bad idea."""
    from utils.utils import ensure_numpy_import
    fixed = ensure_numpy_import("def f(a):\n    return np.copy(a)\n")
    assert fixed.startswith("import numpy as np")
    already = "import numpy as np\ndef f(a):\n    return np.copy(a)\n"
    assert ensure_numpy_import(already) == already


def test_the_objective_is_read_from_the_eval_output_only():
    """The log starts with the candidate's source; a heuristic with the word
    'Error' in a comment must not be discarded because of it."""
    from utils.utils import extract_eval_output, parse_objective
    log = ("# ===== Heuristic Code =====\n"
           "# guard against an Error in the Traceback of long arcs\n"
           "def f(): return 1\n"
           "# ===== Eval Output =====\n"
           "[*] ftv33 (n=34): cost=1286.00 gap=1.234%\n"
           "1.234\n")
    body = extract_eval_output(log)
    assert "Traceback" not in body and "Error" not in body
    assert parse_objective(log) == pytest.approx(1.234)


def test_a_trailing_warning_does_not_break_parsing():
    from utils.utils import parse_objective
    log = "# ===== Eval Output =====\n0.5\nRuntimeWarning: overflow encountered\n"
    assert parse_objective(log) == pytest.approx(0.5)


# ── the offline model ────────────────────────────────────────────────────────

def test_the_stub_answers_all_four_kinds_of_prompt():
    from utils.llm_client.stub import StubClient
    from utils.utils import extract_code, find_txt_block, match_number

    client = StubClient(seed=0)

    directions = client.prompt("e", 'Format your response as a JSON codeblock: '
                                    '{"direction": [...]}')
    assert "direction" in json.loads(extract_code(directions))

    heuristic = client.prompt("e", "implement it as a function named "
                                   "'update_edge_distance'")
    assert "def update_edge_distance(" in extract_code(heuristic)

    optimizer = client.prompt("e", "implement it in Python as a function named "
                                   "'improve_algorithm'")
    code = extract_code(optimizer)
    assert "def improve_algorithm(" in code
    # moh.py rejects any optimizer that never prompts the model.
    assert "language_model.prompt" in code

    count = client.prompt("e", "calculate the number of iterations the outer loop "
                               "will perform")
    assert match_number(find_txt_block(count)) == 1


def test_the_stub_optimizer_is_a_working_optimizer():
    """It has to actually drive the inner loop, or the smoke test proves nothing."""
    from utils.llm_client.stub import StubClient
    from utils.population import Pop
    from utils.utils import extract_code

    client = StubClient(seed=1)
    code = extract_code(client.prompt("e", "a function named 'improve_algorithm'"))
    namespace = {}
    exec(code, namespace)

    pop = Pop(["atsp_gls-50"], size=5)
    pop.save_solution("atsp_gls-50", "seed", "def update_edge_distance(a, b, c):\n"
                                             "    return a", 3.0)
    calls = []

    def utility(solution, idea=None, task=None):
        calls.append(solution)
        return 1.0

    idea, solution, value = namespace["improve_algorithm"](
        pop, utility, client, "function format", "atsp_gls-50")
    assert solution and value is not None
    assert calls, "the optimizer never evaluated anything"


# ── end to end ───────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_a_full_offline_run_writes_every_artefact(tmp_path):
    """Both loops, the evaluation subprocess and the run directory, no API."""
    # The smoke data and budget configs exist for exactly this: two tiny
    # instances and a twentieth of the real engine budget, so a full run of both
    # loops finishes in seconds and needs nothing generated first.
    out = subprocess.run(
        [sys.executable, "main.py", "problem=atsp_gls",
         "llm_client@heu=stub", "llm_client@meta=stub",
         "data=smoke", "evaluation_budget=smoke", "problem.problem_size=[20,40]",
         "n_iterations=1", "pop_size=3", "max_eval_calls=4", "seed_rounds=1"],
        capture_output=True, text=True, cwd=MOH, timeout=1800,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([MOH, ROOT])))
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]

    runs = os.path.join(ROOT, "runs", "llm", "MoH", "atsp_gls-gls")
    latest = max((os.path.join(runs, d) for d in os.listdir(runs)),
                 key=os.path.getmtime)
    for name in ("run.log", "meta.json", "summary.json", "llm_calls.jsonl",
                 "progress.jsonl", "logs/utility.csv", "logs/meta_utility.csv",
                 "evaluations/index.jsonl"):
        assert os.path.isfile(os.path.join(latest, name)), name

    summary = json.load(open(os.path.join(latest, "summary.json"), encoding="utf-8"))
    assert summary["framework"] == "MoH"
    assert summary["llm_calls"] > 0

    meta = json.load(open(os.path.join(latest, "meta.json"), encoding="utf-8"))
    # A key interpolated into the config must never reach disk.
    assert "sk-" not in json.dumps(meta["config"])
