"""Tests for asymmetric safety of ATSP problem engines used by MCTS-AHD."""

import importlib.util
import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AHD = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")

for _p in (AHD, ROOT):
    if _p not in sys.path:
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
    from atsp.data import load_tsplib_atsp
    found = load_tsplib_atsp(os.path.join(ROOT, "data", "raw", "atsp"),
                             best_known_path=os.path.join(ROOT, "data", "raw",
                                                          "atsp", "bestSolutions.txt"),
                             names=["ftv33"])
    assert found, "data/raw/atsp/ftv33.atsp is missing"
    return found[0]


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
    assert cost >= instance.ref_cost


def test_kgls_seed_matches_gls_when_badness_is_cost(instance):
    kgls = _load("t_kgls", "problems/atsp_kgls/kgls.py")

    def badness(dist, tour, penalty, iteration):
        u = np.asarray(tour)
        return dist[u, np.roll(u, -1)].astype(float)

    tour, cost = kgls.knowledge_guided_local_search(
        instance.dist, badness, perturbation_moves=10, iter_limit=20)
    assert sorted(tour) == list(range(instance.n))
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
    kgls = _load("t_kgls4", "problems/atsp_kgls/kgls.py")
    seen = []

    def spy(dist, tour, penalty, iteration):
        seen.append((iteration, float(penalty.sum())))
        u = np.asarray(tour)
        return dist[u, np.roll(u, -1)].astype(float)

    kgls.knowledge_guided_local_search(instance.dist, spy, perturbation_moves=5,
                                       iter_limit=3)
    assert [s[0] for s in seen] == [0, 1, 2]
    assert seen[-1][1] > seen[0][1]


def test_aco_seed_produces_a_valid_tour(instance):
    aco_mod = _load("t_aco", "problems/atsp_aco/aco.py")
    dist = instance.dist.copy()
    np.fill_diagonal(dist, 1.0)
    heu = 1.0 / dist

    colony = aco_mod.ACO(instance.dist, heu, n_ants=5, seed=2024)
    cost = colony.run(3)
    assert np.isfinite(cost) and cost >= instance.ref_cost
    assert sorted(colony.shortest_path.tolist()) == list(range(instance.n))


def test_aco_deposits_only_on_the_traversed_arc(instance):
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
    assert np.all(forward > 0)
    assert np.allclose(backward, 0.0)


def test_aco_rejects_a_non_finite_heuristic(instance):
    aco_mod = _load("t_aco3", "problems/atsp_aco/aco.py")
    heu = np.ones_like(instance.dist)
    heu[0, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        aco_mod.ACO(instance.dist, heu, n_ants=2)
