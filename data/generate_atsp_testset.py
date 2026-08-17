#!/usr/bin/env python
"""Generate synthetic ATSP test instances in TSPLIB format, with proven optima.

TSPLIB ships only 19 ATSP files, so the held-out benchmark in ``data/raw/atsp``
is small and uneven (four ``rbg`` instances carry most of the mean gap, and n
stops at 443). This script writes 70 extra instances in exactly the same file
format, so anything that already reads ``data/raw/atsp`` reads the new folder
unchanged — only the directory name differs::

    NAME / TYPE: ATSP / DIMENSION / EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX / EDGE_WEIGHT_SECTION / EOF
    + bestSolutions.txt beside it, "<name>: <cost>" per line

Ten sizes (n = 50 … 500) x seven generation methods. Four of the methods are
imported from ``solvers/llm/EoH/atsp/data/synthetic.py`` rather than copied, so
training and test instances always come from one definition; three more are
defined here because the existing four leave real gaps in coverage. Measured
over the ladder:

======================  ==========  =====  ===================================
method                  corr        asym   the property nothing else has
======================  ==========  =====  ===================================
uniform                       0.00   0.67  independent directions (rbg regime)
asymmetric_clustered     0.4 - 0.85   0.25  clustered Euclidean, elevation-driven
scheduling_constrained       -0.41   1.58  stage precedence, block penalties
stacker_crane                 0.02   0.53  flat integers, ~7% free arcs
near_symmetric                0.91   0.12  near-symmetric (ftv/kro/ry regime)
shortest_path_metric         -0.04   0.40  obeys the directed triangle inequality
heavy_tailed                  0.02   1.29  log-normal, max/median 125 - 350
======================  ==========  =====  ===================================

``corr`` is corr(d_ij, d_ji), ``asym`` is mean|d_ij - d_ji| / mean(d).
``asymmetric_clustered`` draws its asymmetry strength per instance, which is
why its correlation moves over a range rather than sitting at one value.

Reference costs
---------------
Each instance is solved with **CP-SAT** (``AddCircuit`` — a single-circuit
constraint, i.e. the exact ATSP), so ``bestSolutions.txt`` holds *proven optima*
wherever the proof completes inside the time limit, exactly like the TSPLIB file
it mirrors.

This matters far more than it sounds. An earlier version of this script used the
repo's multi-start GLS + ruin-and-recreate solver as the reference. Measured
against the proven optimum on these instances, that reference was::

    n=50    0.0 - 14.4% high      n=150   1.2 - 75.2% high
    n=100   0.6 - 40.1% high      n=200   2.3 - 80.4% high
    n=500   syn_uni500: 4,001 vs a proven optimum of 1,890  (+112%)

A "reference" that is 112% above the optimum does not measure a heuristic, it
measures the reference. Local search is especially weak on the families with
independent arc costs (``uniform``, ``scheduling_constrained``,
``heavy_tailed``) — which are precisely the families where the assignment
relaxation is tight and CP-SAT proves optimality in seconds.

When a proof does not finish in ``exact_time_limit``, the instance keeps the
best tour found (CP-SAT's incumbent, or the GLS solver's if it happened to be
better) and its ``COMMENT`` line records the interval the optimum is known to
lie in, so a bounded instance is never mistaken for a proven one::

    COMMENT: ... reference=10205 (upper bound, optimum in [9800, 10205], <=4.13% high)

Cost
----
Proof time is dominated by the near-symmetric families; the ones with
independent costs prove quickly even at n=500 (``syn_uni500`` proves in 106s).
Budget a few hours for the full ladder at the default 300s limit, and note that
CP-SAT needs ~5 GB of RAM at n=500 with 8 workers — drop ``workers`` if that is
tight. ``bestSolutions.txt`` is rewritten after every instance, so an
interrupted run resumes where it stopped.

Run:

    pip install ortools                            # the exact solver
    python data/generate_atsp_testset.py           # skips instances already there
    python data/generate_atsp_testset.py --force   # recompute the references
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from solvers.llm.EoH.atsp.data.synthetic import (  # noqa: E402
    generate_instance,
    reference_cost,
)
from solvers.llm.EoH.atsp.data.tsplib import load_best_known, parse_atsp_file  # noqa: E402

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - environment problem
    raise SystemExit(
        "This script needs OR-Tools to prove the optimal tour costs:\n"
        "    pip install ortools\n"
        "Without it the references would be heuristic bounds, which measured up to\n"
        "112% above the true optimum on these instances — see the module docstring.")

OUT_DIR = os.path.join(REPO_ROOT, "data", "synthetic_tsplib", "atsp")

SIZES = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
SEED = 2025

#: seconds CP-SAT may spend proving one instance. Presolve on a 250k-literal
#: model is not interruptible, so at n=500 the wall time per instance can
#: overshoot this by a few minutes.
EXACT_TIME_LIMIT = 300.0

#: CP-SAT search workers. Memory scales with this: ~5 GB at n=500 with 8.
WORKERS = 8


def _near_symmetric(n: int, rng: np.random.Generator, scale: float = 1000.0,
                    asym_frac: float = 0.35, asym_strength: float = 0.5) -> np.ndarray:
    """Euclidean base, then a one-way surcharge on a minority of arcs.

    This is the ``ftv`` / ``kro124p`` / ``ry48p`` regime: corr(d_ij, d_ji) lands
    around 0.9, most of the structure is metric, and 2-opt-style reasoning
    nearly works. ``asymmetric_clustered`` only reaches corr ~0.4, so without
    this family the suite has nothing at the near-symmetric end of TSPLIB.

    It is also the hardest family here to prove optimal — a tight assignment
    bound needs the two directions to disagree, and these barely do.
    """
    coords = rng.random((n, 2))
    base = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2) * scale
    hit = rng.random((n, n)) < asym_frac
    return np.where(hit, base * (1.0 + asym_strength * rng.random((n, n))), base)


def _shortest_path_metric(n: int, rng: np.random.Generator, out_degree: int = 6,
                          max_arc: int = 300) -> np.ndarray:
    """Metric closure of a sparse random digraph.

    A random Hamiltonian cycle is planted first so the digraph is strongly
    connected, extra arcs are added, then Floyd-Warshall closes the matrix. The
    result obeys the *directed* triangle inequality, ``d_ik <= d_ij + d_jk`` for
    every triple — which none of the other six families guarantee. Insertion
    heuristics and Christofides-style bounds only have a ratio on metric
    instances, so a suite without one cannot show whether a heuristic exploits
    metricity or ignores it.
    """
    arcs = np.full((n, n), float(max_arc) * n * 10.0)
    cycle = rng.permutation(n)
    arcs[cycle, np.roll(cycle, -1)] = rng.integers(1, max_arc + 1, size=n)
    for _ in range(max(1, out_degree - 1)):
        arcs[np.arange(n), rng.integers(0, n, size=n)] = rng.integers(1, max_arc + 1, size=n)
    np.fill_diagonal(arcs, 0.0)
    for k in range(n):                      # one numpy pass per k, peak memory n^2
        np.minimum(arcs, arcs[:, k, None] + arcs[None, k, :], out=arcs)
    return arcs


def _heavy_tailed(n: int, rng: np.random.Generator, sigma: float = 1.3,
                  median: float = 200.0) -> np.ndarray:
    """Log-normal costs: a few arcs dominate the whole tour.

    Every other family keeps its costs within a factor of ~20, so an additive
    and a relative view of "expensive" agree. Here they do not — the top
    percentile is a few hundred times the median, and any guide that normalises
    by the mean loses the tail. It is also the realistic regime for mixed-mode
    transport, where one ferry leg sits among road legs.
    """
    return np.exp(rng.normal(np.log(median), sigma, size=(n, n)))


#: family -> (short code used in the instance name, generator or None for the
#: four that already live in solvers/llm/EoH/atsp/data/synthetic.py)
FAMILIES = {
    "uniform": ("uni", None),
    "asymmetric_clustered": ("clu", None),
    "scheduling_constrained": ("sch", None),
    "stacker_crane": ("crn", None),
    "near_symmetric": ("nsy", _near_symmetric),
    "shortest_path_metric": ("met", _shortest_path_metric),
    "heavy_tailed": ("hvy", _heavy_tailed),
}


def _solve_exact(dist: np.ndarray, seed: int, time_limit: float = EXACT_TIME_LIMIT,
                 workers: int = WORKERS) -> tuple[float, bool, float]:
    """Return ``(best_cost, proved_optimal, lower_bound)`` for one instance.

    CP-SAT's ``AddCircuit`` is the exact ATSP: it forces the chosen arcs to form
    one Hamiltonian circuit, so the optimum of this model *is* the optimal tour
    cost — no subtour-elimination loop and no MTZ relaxation gap.

    The repo's GLS solver runs too, purely as insurance: it costs a few seconds
    next to the proof, and taking the better of the two means this can never
    return a worse tour than the heuristic-only version did. In practice CP-SAT
    wins even when it fails to prove — on the two instances that timed out at
    n=150 and n=200 its incumbent was 3.4% and 4.0% below the GLS tour.
    """
    n = int(dist.shape[0])
    costs = np.asarray(dist, dtype=np.float64).round().astype(np.int64)

    model = cp_model.CpModel()
    arcs, objective = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            literal = model.NewBoolVar("")
            arcs.append((i, j, literal))
            objective.append(int(costs[i, j]) * literal)
    model.AddCircuit(arcs)
    model.Minimize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed % (2 ** 31))
    status = solver.Solve(model)

    proved = status == cp_model.OPTIMAL
    bound = float(solver.BestObjectiveBound())
    exact = float(solver.ObjectiveValue()) if status in (cp_model.OPTIMAL,
                                                         cp_model.FEASIBLE) else None
    if proved:
        return exact, True, bound

    # not proved: fall back to the better of CP-SAT's incumbent and the GLS tour
    heuristic = reference_cost(dist, effort="medium" if n <= 100 else "low", seed=seed)
    best = heuristic if exact is None else min(exact, heuristic)
    return float(best), False, bound


def generate_atsp_testset(out_dir: str = OUT_DIR, families: dict = FAMILIES,
                          sizes: tuple = SIZES, seed: int = SEED,
                          exact_time_limit: float = EXACT_TIME_LIMIT,
                          workers: int = WORKERS, force: bool = False) -> str:
    """Write one ``.atsp`` file per (method, size) plus ``bestSolutions.txt``.

    Args:
        out_dir:          destination directory; created if missing.
        families:         ``{name: (short_code, generator or None)}``. A generator
                          takes ``(n, numpy Generator)`` and returns an (n, n) cost
                          matrix; ``None`` means the family comes from the training
                          generator.
        sizes:            cities per instance — one instance per (method, size).
        seed:             base seed. Per-instance seeds are hashed from
                          ``(seed, family, n)``, which keeps them out of the dense
                          low range the training cache uses (``seed * 100_000 +
                          index``) so no test instance can collide with a training
                          one.
        exact_time_limit: seconds CP-SAT may spend proving one instance.
        workers:          CP-SAT search workers; memory scales with this.
        force:            rebuild instances whose file already exists.

    Returns:
        The output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "bestSolutions.txt")

    # the solve is the slow part, so a rerun reuses what is already on disk
    known = load_best_known(best_path) if os.path.exists(best_path) else {}
    best: dict[str, int] = {}
    proved_count = 0
    started = time.perf_counter()

    total = len(families) * len(sizes)
    print(f"Synthetic ATSP test set -> {os.path.relpath(out_dir, REPO_ROOT)}")
    print(f"{total} instances ({len(families)} methods x {len(sizes)} sizes, "
          f"n={sizes[0]}..{sizes[-1]}), CP-SAT limit {exact_time_limit:.0f}s each\n")

    for n in sizes:
        for family, (code, maker) in families.items():
            name = f"syn_{code}{n}"
            path = os.path.join(out_dir, f"{name}.atsp")

            if os.path.exists(path) and not force and name in known:
                best[name] = int(known[name])
                print(f"  {name:<14} n={n:<5} ref={best[name]:>12,.0f}  "
                      f"(kept, --force to rebuild)")
                continue

            # deterministic in (seed, family, n), and far from the training seeds
            key = f"{seed}|{family}|{n}".encode("utf-8")
            inst_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(),
                                       "big") >> 1

            if maker is None:
                dist = generate_instance(family, n, inst_seed)
            else:
                dist = np.round(maker(n, np.random.default_rng(inst_seed)))
                dist = np.maximum(dist, 1.0)        # no free arcs from these three
                np.fill_diagonal(dist, 0.0)

            if not np.all(np.isfinite(dist)) or dist.min() < 0:
                raise ValueError(f"{name}: costs must be finite and non-negative")
            if not np.allclose(dist, np.round(dist)):
                raise ValueError(f"{name}: FULL_MATRIX needs integer costs")
            matrix = dist.astype(np.int64)
            np.fill_diagonal(matrix, 0)

            elapsed = time.perf_counter()
            ref, proved, bound = _solve_exact(dist, inst_seed, exact_time_limit, workers)
            elapsed = time.perf_counter() - elapsed
            proved_count += bool(proved)

            if ref <= 0:
                raise AssertionError(
                    f"{name}: optimal cost is 0, so every ratio-based gap against "
                    f"it is undefined — check the family's free-arc density")

            if proved:
                note = f"reference={ref:.0f} (proven optimal, CP-SAT)"
                label = "optimal"
            else:
                # a very short limit can end with no bound at all, hence the guard
                slack = (f"<={(ref / bound - 1.0) * 100.0:.2f}% high" if bound > 0
                         else "no lower bound found")
                note = (f"reference={ref:.0f} (upper bound, optimum in "
                        f"[{bound:.0f}, {ref:.0f}], {slack})")
                label = f"bound, {slack}"

            # TSPLIB right-aligns the weights in a fixed column and wraps each
            # row at ~80 characters (br17 puts 16 per line, ftv33 puts 6)
            column = len(str(int(matrix.max()))) + 1
            per_line = max(1, 80 // column)
            rows = [
                "".join(f"{int(v):>{column}}" for v in row[start:start + per_line])
                for row in matrix
                for start in range(0, n, per_line)
            ]

            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"NAME: {name}\n"
                         f"TYPE: ATSP\n"
                         f"COMMENT: synthetic ATSP, method={family}, n={n}, "
                         f"seed={inst_seed}, {note}\n"
                         f"DIMENSION: {n}\n"
                         f"EDGE_WEIGHT_TYPE: EXPLICIT\n"
                         f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n"
                         f"EDGE_WEIGHT_SECTION\n")
                fh.write("\n".join(rows))
                fh.write("\nEOF\n")

            # read it back with the repo's own parser: catches a transposed or
            # off-by-one-column write now instead of as an unexplained gap later
            _, reparsed = parse_atsp_file(path)
            expected = matrix.astype(np.float64)
            np.fill_diagonal(expected, 0.0)
            if not np.array_equal(reparsed, expected):
                raise AssertionError(f"{name}: file does not round-trip through "
                                     f"parse_atsp_file")

            best[name] = int(round(ref))
            # rewritten every instance so a Ctrl-C does not throw away the run
            with open(best_path, "w", encoding="utf-8") as fh:
                for key_name in sorted(best):
                    fh.write(f"{key_name}: {best[key_name]}\n")

            print(f"  {name:<14} n={n:<5} ref={ref:>12,.0f}  ({label}, {elapsed:5.1f}s)")

    print(f"\nWrote {len(best)} instances + bestSolutions.txt "
          f"in {time.perf_counter() - started:.0f}s")
    print(f"Proven optimal: {proved_count}/{len(best)}. The rest are upper bounds — "
          f"each file's COMMENT line\ngives the interval its optimum lies in.")
    print(f"Load them with:  load_tsplib_atsp('{os.path.relpath(out_dir, REPO_ROOT)}')")
    return out_dir


if __name__ == "__main__":
    generate_atsp_testset(force="--force" in sys.argv)
