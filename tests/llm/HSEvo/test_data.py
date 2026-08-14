"""Tests for HSEvo ATSP data loading, training splits, and instance resolution."""

import importlib.util
import os
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HSEVO = os.path.join(ROOT, "solvers", "llm", "HSEvo")


def get_hsevo_atsp_utils():
    path = os.path.join(HSEVO, "atsp_utils.py")
    spec = importlib.util.spec_from_file_location("hsevo_atsp_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_training_uses_whole_split():
    atsp_utils = get_hsevo_atsp_utils()
    train_insts = atsp_utils.load_instances("train", 0)
    assert len(train_insts) > 0
    assert max(i.n for i in train_insts) >= 200, "training split must contain large instances (n>=200)"


def test_held_out_instances_not_in_training():
    atsp_utils = get_hsevo_atsp_utils()
    seen = {i.name for i in atsp_utils.load_instances("train")}
    assert "rbg358" not in seen and "rbg443" not in seen, "held-out benchmark instances must not be seen during evolution"


def test_val_instances_loading():
    atsp_utils = get_hsevo_atsp_utils()
    val_insts = atsp_utils.load_instances("val")
    assert len(val_insts) > 0
