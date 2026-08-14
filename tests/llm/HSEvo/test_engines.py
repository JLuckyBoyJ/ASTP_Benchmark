"""Tests for the asymmetric safety of ATSP problem engines used by HSEvo."""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HSEVO = os.path.join(ROOT, "solvers", "llm", "HSEvo")

for _p in (HSEVO, os.path.join(HSEVO, "problems", "atsp_gls"),
           os.path.join(HSEVO, "problems", "atsp_aco"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atsp.data.synthetic import generate_instance, reference_cost
from atsp.data import ATSPInstance
from atsp.engines import is_valid_tour, tour_cost


@pytest.fixture(scope="module")
def instance():
    dist = generate_instance("asymmetric_clustered", 25, seed=11)
    return ATSPInstance(name="c25", dist=dist,
                        ref_cost=reference_cost(dist, effort="low", seed=11),
                        ref_kind="heuristic", source="synthetic")


def test_gls_returns_valid_tour_and_true_cost(instance):
    from gls import guided_local_search
    tour, cost = guided_local_search(instance.dist, instance.dist.copy(),
                                     perturbation_moves=5, iter_limit=10,
                                     time_limit=5.0)
    assert is_valid_tour(tour, instance.n)
    assert cost == pytest.approx(tour_cost(tour, instance.dist))


def test_gls_rejects_malformed_guide(instance):
    from gls import guided_local_search
    with pytest.raises(ValueError):
        guided_local_search(instance.dist, np.zeros((3, 3)), iter_limit=1)
    with pytest.raises(ValueError):
        guided_local_search(instance.dist, instance.dist * np.nan, iter_limit=1)


def test_aco_pheromone_stays_directed(instance):
    from aco import ACO
    dist = instance.dist.copy()
    np.fill_diagonal(dist, 1.0)
    aco = ACO(instance.dist, 1.0 / dist, n_ants=4, seed=0)
    aco.run(3)
    assert not np.allclose(aco.pheromone, aco.pheromone.T), \
        "pheromone was deposited symmetrically on directed graph"


def test_aco_tolerates_infinite_diagonal(instance):
    from aco import ACO
    heu = np.divide(1.0, instance.dist, out=np.full_like(instance.dist, np.inf),
                    where=instance.dist != 0)
    aco = ACO(instance.dist, heu, n_ants=3, seed=0)
    assert np.isfinite(aco.run(2))


def test_aco_rejects_wrongly_shaped_heuristic(instance):
    from aco import ACO
    with pytest.raises(ValueError):
        ACO(instance.dist, np.ones((3, 3)), n_ants=3)
