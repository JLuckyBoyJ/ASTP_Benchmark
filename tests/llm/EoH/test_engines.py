"""Asymmetric-safety tests for the ATSP algorithm engines.

These are the tests that would catch the classic mistake of porting a symmetric
TSP implementation: a move that reverses a tour segment, or a pheromone/penalty
matrix that is silently symmetrised.
"""

import numpy as np
import pytest

from solvers.llm.EoH.atsp.baselines import load_baseline
from solvers.llm.EoH.atsp.data.synthetic import generate_instance, reference_cost
from solvers.llm.EoH.atsp.engines import (
    candidate_lists,
    cheapest_insertion,
    greedy_construct,
    greedy_insertion_tour,
    guided_local_search,
    is_valid_tour,
    local_search,
    local_search_around,
    nearest_neighbour_tour,
    run_aco,
    run_rnr,
    tour_cost,
    tour_edges,
)


@pytest.fixture(scope="module")
def dist():
    return generate_instance("uniform", 40, seed=7)


# ── tour bookkeeping ──────────────────────────────────────────────────────────

def test_tour_cost_uses_direction():
    d = np.array([[0.0, 1.0, 5.0],
                  [9.0, 0.0, 2.0],
                  [3.0, 8.0, 0.0]])
    assert tour_cost([0, 1, 2], d) == 1 + 2 + 3
    assert tour_cost([0, 2, 1], d) == 5 + 8 + 9


def test_tour_edges_are_directed():
    assert tour_edges([2, 0, 1]) == [(2, 0), (0, 1), (1, 2)]


def test_candidate_lists_exclude_self_and_are_bounded(dist):
    cand = candidate_lists(dist, 6)
    assert cand.shape == (40, 6)
    assert all(node not in cand[node] for node in range(40))


def test_constructions_are_valid(dist):
    n = dist.shape[0]
    assert is_valid_tour(nearest_neighbour_tour(dist), n)
    assert is_valid_tour(nearest_neighbour_tour(dist, start=7), n)
    assert is_valid_tour(greedy_insertion_tour(dist), n)


# ── local search ──────────────────────────────────────────────────────────────

def test_local_search_improves_and_reports_the_true_cost(dist):
    cand = candidate_lists(dist, 8)
    start = nearest_neighbour_tour(dist)
    tour, cost = local_search(start, dist, cand)
    assert is_valid_tour(tour, dist.shape[0])
    assert cost < tour_cost(start, dist)
    assert cost == pytest.approx(tour_cost(tour, dist))


def test_local_search_is_a_fixed_point_at_its_optimum(dist):
    cand = candidate_lists(dist, 8)
    tour, cost = local_search(nearest_neighbour_tour(dist), dist, cand)
    again, cost_again = local_search(tour, dist, cand)
    assert cost_again == pytest.approx(cost)
    assert list(again) == list(tour)


def test_or_opt_relocation_is_found():
    """A cheap relocate exists; Or-opt must take it (and not reverse anything)."""
    n = 6
    d = np.full((n, n), 100.0)
    np.fill_diagonal(d, 0.0)
    ring = [0, 1, 2, 3, 4, 5]
    for i in range(n):                      # a cheap directed ring
        d[ring[i], ring[(i + 1) % n]] = 1.0
    tour = [0, 2, 1, 3, 4, 5]               # 1 and 2 swapped -> expensive
    cand = candidate_lists(d, 5)
    improved, cost = local_search(tour, d, cand)
    assert cost == pytest.approx(6.0)
    assert is_valid_tour(improved, n)


def test_local_search_around_only_touches_requested_nodes(dist):
    cand = candidate_lists(dist, 8)
    tour, _ = local_search(nearest_neighbour_tour(dist), dist, cand)
    same, delta = local_search_around(tour, dist, cand, (0, 1))
    assert delta <= 0
    assert is_valid_tour(same, dist.shape[0])


# ── guided local search ───────────────────────────────────────────────────────

def test_gls_beats_or_matches_plain_local_search(dist):
    cand = candidate_lists(dist, 8)
    _, ls_cost = local_search(nearest_neighbour_tour(dist), dist, cand)
    tour, cost, iterations = guided_local_search(
        dist, load_baseline("gls"), time_limit=5.0, ite_max=100,
        n_candidates=8, cand=cand)
    assert iterations >= 1
    assert is_valid_tour(tour, dist.shape[0])
    assert cost == pytest.approx(tour_cost(tour, dist))
    assert cost <= ls_cost + 1e-9


def test_gls_penalties_stay_directed(dist):
    """The engine must not mirror a penalty onto the reverse arc."""
    seen = {}

    def spy(edge_distance, tour, penalties):
        seen["penalties"] = penalties.copy()
        return load_baseline("gls")(edge_distance, tour, penalties)

    guided_local_search(dist, spy, time_limit=5.0, ite_max=6, n_candidates=8)
    penalties = seen["penalties"]
    assert penalties.sum() > 0, "no arc was ever penalised"
    assert not np.allclose(penalties, penalties.T), "penalties were symmetrised"


def test_gls_rejects_a_malformed_matrix(dist):
    with pytest.raises(ValueError):
        guided_local_search(dist, lambda e, t, p: np.full((2, 2), 1.0),
                            time_limit=2.0, ite_max=2, n_candidates=6)
    with pytest.raises(ValueError):
        guided_local_search(dist, lambda e, t, p: e * np.nan,
                            time_limit=2.0, ite_max=2, n_candidates=6)


# ── ACO ───────────────────────────────────────────────────────────────────────

def test_aco_pheromone_is_directed(dist):
    seen = {}

    def spy(pheromone, tours, costs, best_tour, best_cost, rho, it, max_it):
        updated = load_baseline("aco")(pheromone, tours, costs, best_tour,
                                       best_cost, rho, it, max_it)
        seen["updated"] = np.asarray(updated).copy()
        return updated

    cost = run_aco(dist, spy, n_ants=4, iter_max=2, seed=0)
    assert np.isfinite(cost) and cost > 0
    assert not np.allclose(seen["updated"], seen["updated"].T), \
        "Ant System deposited on both directions"


def test_aco_rejects_a_malformed_matrix(dist):
    with pytest.raises(ValueError):
        run_aco(dist, lambda *a: np.zeros((3, 3)), n_ants=3, iter_max=1, seed=0)


# ── ruin and recreate ─────────────────────────────────────────────────────────

def test_cheapest_insertion_keeps_every_node():
    d = generate_instance("uniform", 12, seed=3)
    partial = [0, 1, 2, 3, 4, 5, 6, 7]
    removed = [8, 9, 10, 11]
    tour = cheapest_insertion(partial, removed, d)
    assert sorted(tour) == list(range(12))


def test_rnr_returns_a_real_tour_cost(dist):
    cost = run_rnr(dist, load_baseline("rnr"), iter_max=10, time_limit=3.0,
                   n_candidates=8)
    _, ls_cost = local_search(nearest_neighbour_tour(dist), dist,
                              candidate_lists(dist, 8))
    assert cost <= ls_cost + 1e-9


def test_rnr_rejects_out_of_range_nodes(dist):
    with pytest.raises(ValueError):
        run_rnr(dist, lambda tour, d, k: np.full(k, 10_000), iter_max=3,
                time_limit=2.0, n_candidates=6)


# ── construction ──────────────────────────────────────────────────────────────

def test_greedy_construct_rejects_invalid_choices(dist):
    with pytest.raises(ValueError):
        greedy_construct(dist, lambda c, d, u, m: -1)
    with pytest.raises(ValueError):
        greedy_construct(dist, lambda c, d, u, m: 0)      # already visited


def test_construct_heuristic_cannot_mutate_the_matrix(dist):
    def vandal(current, destination, unvisited, matrix):
        matrix[0, 1] = -1.0                                # must raise
        return int(unvisited[0])

    with pytest.raises(ValueError):
        greedy_construct(dist, vandal)


# ── reference solver ──────────────────────────────────────────────────────────

def test_reference_is_deterministic_and_strong(dist):
    a = reference_cost(dist, effort="low", seed=1)
    b = reference_cost(dist, effort="low", seed=1)
    assert a == b
    _, ls_cost = local_search(nearest_neighbour_tour(dist), dist,
                              candidate_lists(dist, 12))
    assert a <= ls_cost + 1e-9
