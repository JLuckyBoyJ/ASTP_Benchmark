"""MoH data splits, and the size filtering its subtasks depend on."""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOH = os.path.join(ROOT, "solvers", "llm", "MoH")

if MOH not in sys.path:
    sys.path.insert(0, MOH)


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

def test_test_split_is_the_whole_benchmark():
    from atsp.data import resolve_split
    import atsp_utils
    if not os.path.isfile(os.path.join(ROOT, "data", "raw", "atsp", "bestSolutions.txt")):
        pytest.skip("the TSPLIB ATSP benchmark is not present under data/raw/atsp")
    instances = resolve_split(atsp_utils.test_split(), ROOT, log=lambda *a: None)
    assert len(instances) == 19, f"expected 19 TSPLIB ATSP instances, got {len(instances)}"
    assert all(i.ref_kind == "optimal" for i in instances)


def test_train_split_is_the_val_split():
    import atsp_utils
    assert atsp_utils.train_split() == atsp_utils.val_split()


def test_search_split_covers_every_configured_subtask_size():
    """A size named by a problem config but absent from the data config would
    fail at the first evaluation, after the seeding prompts had been paid for."""
    import yaml
    import atsp_utils
    _require_synthetic()
    sizes = set(atsp_utils.available_sizes("val"))
    cfg_dir = os.path.join(ROOT, "configs", "llm", "MoH", "cfg", "problem")
    for name in sorted(os.listdir(cfg_dir)):
        cfg = yaml.safe_load(open(os.path.join(cfg_dir, name), encoding="utf-8"))
        missing = sorted(set(cfg["problem_size"]) - sizes)
        assert not missing, (f"{name} wants sizes {missing}, the default data "
                             f"config has {sorted(sizes)}")


def test_load_instances_filters_by_size():
    import atsp_utils
    _require_synthetic()
    size = atsp_utils.available_sizes("val")[0]
    instances = atsp_utils.load_instances("val", size)
    assert instances and all(i.n == size for i in instances)


def test_a_missing_size_raises_rather_than_falling_back():
    """The size weighting in Eq. (2) is meaningless if a subtask is silently
    scored on the wrong instances, so this must be loud."""
    import atsp_utils
    _require_synthetic()
    with pytest.raises(ValueError, match="no val instance of size 12345"):
        atsp_utils.load_instances("val", 12345)


def test_the_default_split_holds_out_the_whole_benchmark(monkeypatch):
    import importlib
    monkeypatch.setenv("ATSP_DATA", "synthetic")
    import atsp_utils
    importlib.reload(atsp_utils)
    assert atsp_utils.trained_on() == set(), (
        "the default synthetic split must not name any TSPLIB instance")
