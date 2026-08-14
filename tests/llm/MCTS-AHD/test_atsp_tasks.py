"""Tests for MCTS-AHD retargeted to ATSP.

Nothing here calls an LLM or costs anything. The tests cover the four places a
port like this actually breaks:

* the **engines** — do the ATSP tasks run, and do they respect asymmetry?
* the **contract** — does every ``eval.py`` print a parseable objective as its
  last line, which is the only thing ``problem_adapter`` reads?
* the **prompts** — does every task have the four files, does the signature
  parse, and does the seed function in the prompt match the one in ``gpt.py``?
* the **search** — do the MCTS primitives (UCT, backpropagation, progressive
  widening, response parsing) behave, including in the degenerate cases that
  crashed the released code?

Run with::

    pytest tests/llm/MCTS-AHD -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AHD = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")
TASKS = ("atsp_constructive", "atsp_gls", "atsp_kgls", "atsp_aco")
IMPROVEMENT_TASKS = ("atsp_gls", "atsp_kgls")

if "atsp_utils" in sys.modules:
    del sys.modules["atsp_utils"]
for _p in (AHD, ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


def _load(name: str, relative: str):
    path = os.path.join(AHD, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instance():
    """One small TSPLIB ATSP instance with a proven optimum."""
    from atsp.data import load_tsplib_atsp
    found = load_tsplib_atsp(os.path.join(ROOT, "data", "raw", "atsp"),
                             best_known_path=os.path.join(ROOT, "data", "raw",
                                                          "atsp", "bestSolutions.txt"),
                             names=["ftv33"])
    assert found, "data/raw/atsp/ftv33.atsp is missing"
    return found[0]


# ── prompts ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_prompt_files_exist(task):
    for name in ("func_signature.txt", "func_desc.txt", "seed_func.txt",
                 "external_knowledge.txt"):
        path = os.path.join(AHD, "prompts", task, name)
        assert os.path.isfile(path), f"missing {task}/{name}"
        assert os.path.getsize(path) > 0, f"empty {task}/{name}"


@pytest.mark.parametrize("task", TASKS)
def test_signature_parses_like_problem_adapter(task):
    """`problem_adapter.Prompts` parses the signature with this exact regex."""
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
    """The seed shipped in gpt.py and the one in the prompt must agree.

    They are two copies of the same baseline — the prompt's is what the model
    is shown, gpt.py's is what runs before the first LLM call and what
    `smoke.sh` scores. A drift between them would silently make the reported
    baseline a different heuristic from the advertised one.
    """
    import ast

    prompt = open(os.path.join(AHD, "prompts", task, "seed_func.txt"),
                  encoding="utf-8").read()
    shipped = open(os.path.join(AHD, "problems", task, "gpt.py"),
                   encoding="utf-8").read()

    def body(text):
        """The function's statements, ignoring its name, docstring and imports."""
        tree = ast.parse(text)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assert len(funcs) == 1, f"{task}: expected exactly one function, got {len(funcs)}"
        statements = funcs[0].body
        if (statements and isinstance(statements[0], ast.Expr)
                and isinstance(statements[0].value, ast.Constant)
                and isinstance(statements[0].value.value, str)):
            statements = statements[1:]      # drop the docstring
        return [ast.dump(node) for node in statements]

    assert body(prompt) == body(shipped), (
        f"{task}: prompts/seed_func.txt and problems/gpt.py have drifted apart")


# ── engines ───────────────────────────────────────────────────────────────────

def test_matrix_is_actually_asymmetric(instance):
    d = instance.dist
    assert not np.allclose(d, d.T), "ftv33 should be asymmetric"
    assert np.allclose(np.diag(d), 0.0)


def test_gls_seed_beats_nearest_neighbour(instance):
    from atsp.engines.tour import nearest_neighbour_tour, tour_cost
    gls = _load("t_gls", "problems/atsp_gls/gls.py")

    d = instance.dist
    nn = tour_cost(nearest_neighbour_tour(d), d)

    guide = d + 0.05 * (d - np.min(np.where(np.eye(instance.n, dtype=bool), np.inf, d),
                                   axis=1, keepdims=True))
    _, cost = gls.guided_local_search(d, guide, perturbation_moves=10, iter_limit=20)
    assert cost < nn
    assert cost >= instance.ref_cost, "cannot beat a proven optimum"


def test_kgls_seed_matches_gls_when_badness_is_cost(instance):
    """With badness = arc cost, KGLS *is* guided local search."""
    kgls = _load("t_kgls", "problems/atsp_kgls/kgls.py")

    def badness(dist, tour, penalty, iteration):
        u = np.asarray(tour)
        return dist[u, np.roll(u, -1)].astype(float)

    tour, cost = kgls.knowledge_guided_local_search(
        instance.dist, badness, perturbation_moves=10, iter_limit=20)
    assert sorted(tour) == list(range(instance.n)), "not a valid tour"
    assert cost >= instance.ref_cost


def test_kgls_rejects_a_wrong_shape(instance):
    kgls = _load("t_kgls2", "problems/atsp_kgls/kgls.py")

    def bad(dist, tour, penalty, iteration):
        return np.ones((len(tour), 2))

    with pytest.raises(ValueError, match="one value per tour arc"):
        kgls.knowledge_guided_local_search(instance.dist, bad, iter_limit=1)


def test_kgls_rejects_non_finite(instance):
    kgls = _load("t_kgls3", "problems/atsp_kgls/kgls.py")

    def bad(dist, tour, penalty, iteration):
        out = np.ones(len(tour))
        out[0] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        kgls.knowledge_guided_local_search(instance.dist, bad, iter_limit=1)


def test_kgls_sees_the_penalty_history(instance):
    """The rule is handed live state, which is the whole point of the task."""
    kgls = _load("t_kgls4", "problems/atsp_kgls/kgls.py")
    seen = []

    def spy(dist, tour, penalty, iteration):
        seen.append((iteration, float(penalty.sum())))
        u = np.asarray(tour)
        return dist[u, np.roll(u, -1)].astype(float)

    kgls.knowledge_guided_local_search(instance.dist, spy, perturbation_moves=5,
                                       iter_limit=3)
    assert [s[0] for s in seen] == [0, 1, 2]
    assert seen[-1][1] > seen[0][1], "penalty counters never reached the rule"


def test_aco_seed_produces_a_valid_tour(instance):
    """The seed desirability 1/d must drive the colony to a legal tour."""
    aco_mod = _load("t_aco", "problems/atsp_aco/aco.py")
    dist = instance.dist.copy()
    np.fill_diagonal(dist, 1.0)
    heu = 1.0 / dist

    colony = aco_mod.ACO(instance.dist, heu, n_ants=5, seed=2024)
    cost = colony.run(3)
    assert np.isfinite(cost) and cost >= instance.ref_cost
    assert sorted(colony.shortest_path.tolist()) == list(range(instance.n))


def test_aco_deposits_only_on_the_traversed_arc(instance):
    """ATSP pheromone is directed; depositing on (v, u) too would be wrong."""
    aco_mod = _load("t_aco2", "problems/atsp_aco/aco.py")
    dist = instance.dist.copy()
    np.fill_diagonal(dist, 1.0)
    colony = aco_mod.ACO(instance.dist, 1.0 / dist, n_ants=1, seed=0)

    before = colony.pheromone.copy()
    path = np.arange(instance.n)
    colony.update_pheromone([path], [1000.0])
    delta = colony.pheromone - before * colony.decay
    forward = delta[path, np.roll(path, -1)]
    backward = delta[np.roll(path, -1), path]
    assert np.all(forward > 0), "the traversed arcs got no pheromone"
    assert np.allclose(backward, 0.0), "the reverse arcs were reinforced too"


def test_aco_rejects_a_non_finite_heuristic(instance):
    aco_mod = _load("t_aco3", "problems/atsp_aco/aco.py")
    heu = np.ones_like(instance.dist)
    heu[0, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        aco_mod.ACO(instance.dist, heu, n_ants=2)


# ── the evaluation contract ───────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_eval_prints_a_parseable_objective_last(task, tmp_path):
    """`problem_adapter` does float(stdout.split('\\n')[-2]) — nothing else."""
    env = dict(os.environ)
    # One small instance keeps the test quick; the split itself is covered by
    # test_train_split_is_shared_with_reevo below.
    env["ATSP_TRAIN_SPLIT"] = "tsplib"
    script = os.path.join(AHD, "problems", task, "eval.py")
    proc = subprocess.run([sys.executable, "-u", script, "0", AHD, "val"],
                          capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    objective = float(proc.stdout.split("\n")[-2])
    assert objective == objective and objective >= 0


# ── data split ────────────────────────────────────────────────────────────────

def test_train_split_is_shared_with_reevo():
    """The comparison is only meaningful if the fitness function is the same."""
    import atsp_utils as ahd_utils

    reevo_path = os.path.join(ROOT, "solvers", "llm", "ReEvo", "atsp_utils.py")
    if not os.path.isfile(reevo_path):
        pytest.skip("ReEvo solver is not present")
    spec = importlib.util.spec_from_file_location("reevo_atsp_utils", reevo_path)
    reevo_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reevo_mod)

    assert ahd_utils.train_split() == reevo_mod.TRAIN_SPLIT, (
        f"MCTS-AHD train split ({ahd_utils.train_split()}) differs from ReEvo ({reevo_mod.TRAIN_SPLIT})"
    )


def test_test_split_is_the_whole_benchmark():
    from atsp.data import resolve_split
    import atsp_utils as ahd_utils
    instances = resolve_split(ahd_utils.test_split(), ROOT, log=lambda *a: None)
    assert len(instances) == 19, f"expected 19 TSPLIB ATSP instances, got {len(instances)}"
    assert all(i.ref_kind == "optimal" for i in instances)


# ── the search itself ─────────────────────────────────────────────────────────

def test_uct_survives_a_degenerate_value_range():
    """The released code divided by q_max - q_min with no guard."""
    from source.mcts import MCTS, MCTSNode

    mcts = MCTS("Root")
    node = MCTSNode("a", "code", obj=5.0, parent=mcts.root, depth=1, visit=1, Q=-5.0)
    mcts.root.add_child(node)
    mcts.backpropagate(node)
    mcts.q_min = mcts.q_max = -5.0            # force the degenerate case
    assert np.isfinite(mcts.uct(node, 1.0))


def test_node_repr_does_not_raise():
    """The released __repr__ referenced an attribute no node has."""
    from source.mcts import MCTSNode
    node = MCTSNode("idea", "code", obj=1.0, depth=1, visit=1, Q=-1.0, action="m1")
    assert "m1" in repr(node)


def test_backpropagation_lifts_the_best_child():
    """Eq. (6): a parent's Q is the best Q anywhere in its subtree."""
    from source.mcts import MCTS, MCTSNode

    mcts = MCTS("Root")
    parent = MCTSNode("p", "p", obj=10.0, parent=mcts.root, depth=1, visit=1, Q=-10.0)
    mcts.root.add_child(parent)
    mcts.backpropagate(parent)

    child = MCTSNode("c", "c", obj=2.0, parent=parent, depth=2, visit=1, Q=-2.0)
    parent.add_child(child)
    mcts.backpropagate(child)

    assert parent.Q == -2.0, "a good child must lift its parent"
    assert mcts.root.Q == -2.0


def test_inferior_nodes_are_kept():
    """The paper's central claim: nothing is discarded for scoring badly."""
    from source.mcts import MCTS, MCTSNode

    mcts = MCTS("Root")
    for obj in (1.0, 99.0):
        node = MCTSNode(f"n{obj}", "code", obj=obj, parent=mcts.root, depth=1,
                        visit=1, Q=-obj)
        mcts.root.add_child(node)
        mcts.backpropagate(node)
    assert len(mcts.root.children) == 2
    assert any(n.Q == -99.0 for n in mcts.root.children)


def test_progressive_widening_condition():
    """Eq. (4): a new child is added when floor(N^alpha) >= |children|."""
    from source.mcts import MCTS, MCTSNode

    mcts = MCTS("Root", alpha=0.5)
    node = MCTSNode("n", "code", obj=1.0, parent=mcts.root, depth=1, visit=9, Q=-1.0)
    node.children = [1, 2]                 # placeholders; only the count matters
    assert int(node.visits ** mcts.alpha) > len(node.children)   # 3 > 2 -> widen
    node.children = [1, 2, 3]
    assert not int(node.visits ** mcts.alpha) > len(node.children)


# ── configuration, not constants ──────────────────────────────────────────────

@pytest.mark.parametrize("name", ["tsplib", "synthetic", "mcts_ahd_native"])
def test_every_data_config_declares_three_splits(name):
    import atsp_utils
    config = atsp_utils.load_config("data", name)
    for mood in ("train", "val", "test"):
        assert config.get(mood), f"{name}.yaml has no {mood} split"


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_an_evaluation_budget(task):
    import atsp_utils
    for mood in ("train", "val", "test"):
        settings = atsp_utils.budget(task, mood)
        assert isinstance(settings, dict)


@pytest.mark.parametrize("task", IMPROVEMENT_TASKS)
def test_training_budget_tracks_the_benchmark(task):
    """`iter_limit: auto` must resolve to something, and to more on a bigger n."""
    import atsp_utils
    settings = atsp_utils.budget(task, "train")
    assert settings["iter_limit"] == "auto"
    small = atsp_utils.resolve_iter_limit("auto", 50)
    large = atsp_utils.resolve_iter_limit("auto", 400)
    assert small > large > 0, "a bigger instance must get fewer iterations"
    assert atsp_utils.resolve_iter_limit(1000, 50) == 1000


def test_gls_and_kgls_get_the_same_engine_budget():
    """The two improvement tasks must differ only in the design space."""
    import atsp_utils
    for mood in ("train", "val", "test"):
        gls = atsp_utils.budget("atsp_gls", mood)
        kgls = atsp_utils.budget("atsp_kgls", mood)
        assert gls == kgls, (
            f"atsp_gls and atsp_kgls have different {mood} budgets: {gls} vs {kgls}; "
            "any difference between the tasks would then be confounded")


def test_data_config_selection_honours_the_environment(monkeypatch):
    import atsp_utils
    monkeypatch.setenv("ATSP_DATA", "synthetic")
    assert atsp_utils.data_config_name() == "synthetic"
    monkeypatch.delenv("ATSP_DATA")
    monkeypatch.setenv("ATSP_TRAIN_SPLIT", "synth")     # legacy alias
    assert atsp_utils.data_config_name() == "synthetic"
    monkeypatch.delenv("ATSP_TRAIN_SPLIT")
    assert atsp_utils.data_config_name() == "tsplib"


def test_solver_registry_lists_every_task():
    import yaml
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


def test_no_dead_modules_in_the_solver():
    """Every .py under the solver must be reachable, or explicitly excused."""
    import glob
    excused = {
        # written by the search on every evaluation, imported as bare `gpt`
        "problems/atsp_constructive/gpt.py", "problems/atsp_gls/gpt.py",
        "problems/atsp_kgls/gpt.py", "problems/atsp_aco/gpt.py",
        # instantiated by Hydra from cfg/llm_client/*.yaml, never imported
        "utils/llm_client/litellm.py", "utils/llm_client/stub.py",
    }
    sources = {os.path.relpath(p, AHD)
               for p in glob.glob(os.path.join(AHD, "**", "*.py"), recursive=True)}
    blob = "\n".join(open(os.path.join(AHD, s), encoding="utf-8").read()
                      for s in sources)
    for rel in sorted(sources):
        if rel in excused or rel.endswith("__init__.py"):
            continue
        stem = os.path.splitext(os.path.basename(rel))[0]
        assert stem in blob.replace(rel, ""), (
            f"{rel} is never imported by anything — delete it or excuse it here")


def test_config_exposes_the_papers_defaults():
    import yaml
    path = os.path.join(ROOT, "configs", "llm", "MCTS-AHD", "cfg", "config.yaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    assert cfg["init_pop_size"] == 4            # N_I
    assert cfg["pop_size"] == 10                # |E|
    assert cfg["exploration_constant"] == 0.1   # lambda_0
    assert cfg["progressive_widening_alpha"] == 0.5
    assert cfg["expansion_children"] == 2       # k, giving 2k+2 children
    assert cfg["data"] in ("synthetic", "tsplib")


@pytest.mark.parametrize("task", TASKS)
def test_problem_config_matches_the_task(task):
    import yaml
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


# ── response parsing ──────────────────────────────────────────────────────────

class _FakePrompts:
    def get_task(self): return "task"
    def get_func_name(self): return "heuristics"
    def get_func_inputs(self): return ["distance_matrix"]
    def get_func_outputs(self): return ["heuristics_matrix"]
    def get_inout_inf(self): return "desc"
    def get_other_inf(self): return ""


def _evolution():
    from source.evolution import Evolution
    return Evolution("endpoint", "key", None, False, _FakePrompts(),
                     use_local_llm=False, url=None)


def test_extracts_a_fenced_code_block():
    """gpt-4o-mini fences its code; the released regex swallowed the fence."""
    evol = _evolution()
    response = ("{Penalise long arcs.}\n\n```python\nimport numpy as np\n"
                "def heuristics(distance_matrix):\n    return distance_matrix\n```")
    code = evol._extract_code(response)
    assert code is not None and "```" not in code
    exec(compile(code, "<t>", "exec"), {"np": np})


def test_falls_back_to_the_upstream_regex():
    evol = _evolution()
    response = ("{An idea.}\nimport numpy as np\n"
                "def heuristics(distance_matrix):\n    return")
    code = evol._extract_code(response)
    assert code is not None and code.strip().endswith("heuristics_matrix")


def test_missing_braces_do_not_crash():
    """The released code called .group(1) on a possibly-None match."""
    evol = _evolution()
    response = ("Here is the function.\n```python\n"
                "def heuristics(distance_matrix):\n    return distance_matrix\n```")
    assert evol._extract_code(response) is not None
    assert evol._extract_thought(response, "x")


def test_a_response_with_no_code_is_reported_not_looped():
    evol = _evolution()

    class _NoCode:
        def get_response(self, prompt, temp=1.0):
            return "I would rather not."

    evol.interface_llm = _NoCode()
    with pytest.raises(ValueError, match="no usable function"):
        evol._get_alg("prompt", n_retry=2)


# ── logging ───────────────────────────────────────────────────────────────────

def test_actions_are_recognised_from_their_prompts():
    from utils.atsp_logging import _infer_action

    evol = _evolution()
    indiv = {"algorithm": "idea", "code": "def heuristics(d): return d",
             "objective": 1.0}
    cases = {
        "i1": evol.get_prompt_i1(),
        "e1": evol.get_prompt_e1([indiv, indiv]),
        "e2": evol.get_prompt_e2([indiv, indiv]),
        "m1": evol.get_prompt_m1(indiv),
        "m2": evol.get_prompt_m2(indiv),
        "s1": evol.get_prompt_s1([indiv, indiv]),
        "thought_align": evol.get_prompt_refine("code", "idea"),
    }
    for expected, prompt in cases.items():
        assert _infer_action([{"content": prompt}]) == expected, expected


def test_tree_serialisation_round_trips(tmp_path):
    from source.mcts import MCTS, MCTSNode
    from utils.atsp_logging import write_mcts_tree

    mcts = MCTS("Root")
    node = MCTSNode("an idea", "def f(): pass", obj=3.0, parent=mcts.root, depth=1,
                    visit=1, Q=-3.0, raw_info={"objective": 3.0}, action="i1")
    mcts.root.add_child(node)
    mcts.backpropagate(node)

    path = write_mcts_tree(str(tmp_path), mcts.root, {"problem": "atsp_gls"})
    payload = json.load(open(path, encoding="utf-8"))
    assert payload["problem"] == "atsp_gls"
    assert payload["tree"]["children"][0]["action"] == "i1"
    assert payload["tree"]["children"][0]["objective"] == 3.0


def test_dotenv_does_not_override_a_real_variable(tmp_path, monkeypatch):
    from utils.atsp_logging import load_dotenv

    env_file = tmp_path / "envs" / ".env"
    env_file.parent.mkdir()
    env_file.write_text('OPENAI_API_KEY="from-file"\nexport OTHER=42\n')
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.delenv("ENV_FILE", raising=False)

    loaded = load_dotenv(str(tmp_path))
    assert loaded == [str(env_file)]
    assert os.environ["OPENAI_API_KEY"] == "from-environment"
    assert os.environ["OTHER"] == "42"


# ── benchmarking ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_benchmark_runner_scores_the_seed(task, instance, tmp_path):
    from evaluation.llm.mcts_ahd import (
        run_benchmark, seed_heuristic, write_results, TASKS as BENCH_TASKS)

    assert task in BENCH_TASKS
    from evaluation.llm.mcts_ahd import default_params
    params = dict(default_params(task))
    if "iter_limit" in params:                      # keep the test quick
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
