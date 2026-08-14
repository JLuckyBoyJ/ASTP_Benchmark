"""Tests for MCTS-AHD data splits and instance resolution."""

import importlib.util
import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AHD = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")

if AHD not in sys.path:
    sys.path.insert(0, AHD)


def test_train_split_is_shared_with_reevo():
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
