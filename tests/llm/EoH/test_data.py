"""Tests for the ATSP data layer used by EoH: synthetic families and splits."""

import os

import numpy as np
import pytest

from solvers.llm.EoH.atsp.data import (
    FAMILIES,
    generate_instance,
    load_dataset,
    resolve_split,
    save_dataset,
)
from solvers.llm.EoH.atsp.data.synthetic import (
    EFFORT_LEVELS,
    dataset_filename,
    generate_dataset,
    reference_cost,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.mark.parametrize("family", FAMILIES)
def test_families_are_deterministic_and_asymmetric(family):
    a = generate_instance(family, 20, seed=7)
    b = generate_instance(family, 20, seed=7)
    assert np.array_equal(a, b)
    assert a.shape == (20, 20)
    assert np.all(np.diag(a) == 0)
    assert np.all(a >= 0)
    assert not np.allclose(a, a.T), f"{family} must be asymmetric"


def test_different_seeds_give_different_instances():
    assert not np.array_equal(generate_instance("uniform", 15, seed=1),
                              generate_instance("uniform", 15, seed=2))


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        generate_instance("euclidean", 10, seed=1)


@pytest.mark.parametrize("effort", sorted(EFFORT_LEVELS))
def test_reference_effort_levels_run(effort):
    dist = generate_instance("uniform", 15, seed=4)
    cost = reference_cost(dist, effort=effort, seed=4)
    assert np.isfinite(cost) and cost > 0


def test_reference_effort_is_monotone_in_quality():
    dist = generate_instance("uniform", 30, seed=9)
    assert reference_cost(dist, effort="medium", seed=9) <= \
           reference_cost(dist, effort="low", seed=9) + 1e-9


def test_unknown_effort_is_rejected():
    with pytest.raises(ValueError):
        reference_cost(generate_instance("uniform", 10, seed=1), effort="max")


def test_dataset_roundtrip(tmp_path):
    instances = generate_dataset("uniform", 12, 2, seed=3, effort="low")
    path = str(tmp_path / dataset_filename("uniform", 12, 2, 3))
    save_dataset(path, instances)
    loaded = load_dataset(path)

    assert len(loaded) == 2
    assert loaded[0].name == instances[0].name
    assert np.array_equal(loaded[0].dist, instances[0].dist)
    assert loaded[0].ref_cost == pytest.approx(instances[0].ref_cost)
    assert loaded[0].ref_kind == "heuristic"
    assert loaded[0].meta["family"] == "uniform"


def test_gap_is_relative_to_the_reference():
    instance = generate_dataset("uniform", 10, 1, seed=5, effort="low")[0]
    assert instance.gap(instance.ref_cost) == pytest.approx(0.0)
    assert instance.gap(2 * instance.ref_cost) == pytest.approx(100.0)


def test_resolve_split_caches_synthetic_sets(tmp_path):
    spec = {"source": "synthetic", "family": "uniform", "size": 12,
            "count": 2, "seed": 11, "effort": "low",
            "path": str(tmp_path / "train.npz")}
    first = resolve_split(spec, ROOT, log=lambda *_: None)
    assert os.path.exists(spec["path"])
    second = resolve_split(spec, ROOT, log=lambda *_: None)   # now from cache
    assert [i.name for i in first] == [i.name for i in second]
    assert np.array_equal(first[0].dist, second[0].dist)


def test_resolve_split_reads_the_tsplib_test_set():
    instances = resolve_split({"source": "tsplib", "dir": "data/raw/atsp",
                               "best_known": "data/raw/atsp/bestSolutions.txt",
                               "max_n": 50},
                              ROOT, log=lambda *_: None)
    assert instances and all(ins.n <= 50 for ins in instances)
    assert all(ins.ref_kind == "optimal" for ins in instances)
    assert instances == sorted(instances, key=lambda i: (i.n, i.name))


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        resolve_split({"source": "kaggle"}, ROOT, log=lambda *_: None)
