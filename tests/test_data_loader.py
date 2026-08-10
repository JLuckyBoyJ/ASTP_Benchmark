"""Integrity checks for the benchmark corpus in `data/raw/atsp`.

Solver-specific data tests live in `tests/llm/EoH/test_data.py`; this file is
about the raw dataset itself.
"""

import os

import numpy as np

from solvers.llm.EoH.atsp.data import load_best_known, load_tsplib_atsp, parse_atsp_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATSP_DIR = os.path.join(ROOT, "data", "raw", "atsp")

EXPECTED = {
    "br17": (17, 39), "ftv33": (34, 1286), "ftv35": (36, 1473), "ftv38": (39, 1530),
    "p43": (43, 5620), "ftv44": (45, 1613), "ftv47": (48, 1776), "ry48p": (48, 14422),
    "ft53": (53, 6905), "ftv55": (56, 1608), "ftv64": (65, 1839), "ft70": (70, 38673),
    "ftv70": (71, 1950), "kro124p": (100, 36230), "ftv170": (171, 2755),
    "rbg323": (323, 1326), "rbg358": (358, 1163), "rbg403": (403, 2465),
    "rbg443": (443, 2720),
}


def test_all_nineteen_instances_are_present():
    files = {f[:-5] for f in os.listdir(ATSP_DIR) if f.endswith(".atsp")}
    assert files == set(EXPECTED)


def test_dimensions_and_optima_match_tsplib():
    instances = {ins.name: ins for ins in load_tsplib_atsp(ATSP_DIR)}
    assert set(instances) == set(EXPECTED)
    for name, (n, opt) in EXPECTED.items():
        assert instances[name].n == n, f"{name}: wrong dimension"
        assert instances[name].ref_cost == opt, f"{name}: wrong optimum"
        assert instances[name].ref_kind == "optimal"


def test_matrices_are_square_asymmetric_and_non_negative():
    for instance in load_tsplib_atsp(ATSP_DIR):
        dist = instance.dist
        assert dist.shape == (instance.n, instance.n)
        assert np.all(np.diag(dist) == 0), f"{instance.name}: diagonal not normalised"
        assert np.all(dist >= 0), f"{instance.name}: negative arc"
        assert not np.allclose(dist, dist.T), f"{instance.name}: not asymmetric"


def test_wrapped_matrix_rows_are_parsed_correctly():
    """br17 wraps its rows across lines and uses 9999 on the diagonal."""
    _, dist = parse_atsp_file(os.path.join(ATSP_DIR, "br17.atsp"))
    assert dist.shape == (17, 17)
    assert dist[0, 1] == 3 and dist[3, 2] == 74 and dist[2, 3] == 72
    assert dist.max() < 9999


def test_best_known_file_covers_every_instance():
    best = load_best_known(os.path.join(ATSP_DIR, "bestSolutions.txt"))
    assert best["ftv33"] == 1286
    # the file also lists ftv90..ftv160, whose .atsp files are not shipped
    assert len(best) >= len(EXPECTED)


def test_size_filters_work():
    small = load_tsplib_atsp(ATSP_DIR, max_n=50)
    assert small and all(ins.n <= 50 for ins in small)
    large = load_tsplib_atsp(ATSP_DIR, min_n=300)
    assert {ins.name for ins in large} == {"rbg323", "rbg358", "rbg403", "rbg443"}
