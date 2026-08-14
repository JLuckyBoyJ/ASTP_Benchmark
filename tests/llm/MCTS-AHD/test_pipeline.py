"""Tests for MCTS primitives, progressive widening, backpropagation, logging, and response parsing in MCTS-AHD."""

import json
import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AHD = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")

for _p in (AHD, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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


def test_uct_survives_a_degenerate_value_range():
    from source.mcts import MCTS, MCTSNode

    mcts = MCTS("Root")
    node = MCTSNode("a", "code", obj=5.0, parent=mcts.root, depth=1, visit=1, Q=-5.0)
    mcts.root.add_child(node)
    mcts.backpropagate(node)
    mcts.q_min = mcts.q_max = -5.0
    assert np.isfinite(mcts.uct(node, 1.0))


def test_node_repr_does_not_raise():
    from source.mcts import MCTSNode
    node = MCTSNode("idea", "code", obj=1.0, depth=1, visit=1, Q=-1.0, action="m1")
    assert "m1" in repr(node)


def test_backpropagation_lifts_the_best_child():
    from source.mcts import MCTS, MCTSNode

    mcts = MCTS("Root")
    parent = MCTSNode("p", "p", obj=10.0, parent=mcts.root, depth=1, visit=1, Q=-10.0)
    mcts.root.add_child(parent)
    mcts.backpropagate(parent)

    child = MCTSNode("c", "c", obj=2.0, parent=parent, depth=2, visit=1, Q=-2.0)
    parent.add_child(child)
    mcts.backpropagate(child)

    assert parent.Q == -2.0
    assert mcts.root.Q == -2.0


def test_inferior_nodes_are_kept():
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
    from source.mcts import MCTS, MCTSNode

    mcts = MCTS("Root", alpha=0.5)
    node = MCTSNode("n", "code", obj=1.0, parent=mcts.root, depth=1, visit=9, Q=-1.0)
    node.children = [1, 2]
    assert int(node.visits ** mcts.alpha) > len(node.children)
    node.children = [1, 2, 3]
    assert not int(node.visits ** mcts.alpha) > len(node.children)


def test_extracts_a_fenced_code_block():
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
