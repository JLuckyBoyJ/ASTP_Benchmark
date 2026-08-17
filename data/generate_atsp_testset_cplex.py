#!/usr/bin/env python
"""Generate synthetic ATSP test instances in TSPLIB format, with proven optima using CPLEX IBM.

TSPLIB ships only 19 ATSP files, so the held-out benchmark in ``data/raw/atsp``
is small and uneven (four ``rbg`` instances carry most of the mean gap, and n
stops at 443). This script writes 70 extra instances in exactly the same file
format, storing them in ``data/synthetic_tsplib_cplex/atsp`` so solvers can
benchmark against CPLEX-proven references:

    NAME / TYPE: ATSP / DIMENSION / EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX / EDGE_WEIGHT_SECTION / EOF
    + bestSolutions.txt beside it, "<name>: <cost>" per line

Reference costs
---------------
Each instance is solved with **IBM CPLEX** (MILP with Lazy Constraint Callback
for Subtour Elimination), so ``bestSolutions.txt`` holds *proven optima*
wherever the proof completes inside the time limit.

Run:

    pip install docplex cplex                        # CPLEX Python packages
    python data/generate_atsp_testset_cplex.py        # skips instances already there
    python data/generate_atsp_testset_cplex.py --force# recompute the references
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
    import cplex
    from cplex.callbacks import LazyConstraintCallback
    HAS_CPLEX = True
except ImportError:
    HAS_CPLEX = False
    try:
        import docplex.mp.model as docplex_model
    except ImportError:
        raise SystemExit(
            "This script needs IBM ILOG CPLEX (docplex / cplex) to prove optimal tour costs:\n"
            "    pip install docplex cplex\n"
            "See module docstring for details."
        )

OUT_DIR = os.path.join(REPO_ROOT, "data", "synthetic_tsplib_cplex", "atsp")

SIZES = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
SEED = 2025

EXACT_TIME_LIMIT = 300.0
WORKERS = 8


def _near_symmetric(n: int, rng: np.random.Generator, scale: float = 1000.0,
                    asym_frac: float = 0.35, asym_strength: float = 0.5) -> np.ndarray:
    """Euclidean base, then a one-way surcharge on a minority of arcs."""
    coords = rng.random((n, 2))
    base = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2) * scale
    hit = rng.random((n, n)) < asym_frac
    return np.where(hit, base * (1.0 + asym_strength * rng.random((n, n))), base)


def _shortest_path_metric(n: int, rng: np.random.Generator, out_degree: int = 6,
                          max_arc: int = 300) -> np.ndarray:
    """Metric closure of a sparse random digraph."""
    arcs = np.full((n, n), float(max_arc) * n * 10.0)
    cycle = rng.permutation(n)
    arcs[cycle, np.roll(cycle, -1)] = rng.integers(1, max_arc + 1, size=n)
    for _ in range(max(1, out_degree - 1)):
        arcs[np.arange(n), rng.integers(0, n, size=n)] = rng.integers(1, max_arc + 1, size=n)
    np.fill_diagonal(arcs, 0.0)
    for k in range(n):
        np.minimum(arcs, arcs[:, k, None] + arcs[None, k, :], out=arcs)
    return arcs


def _heavy_tailed(n: int, rng: np.random.Generator, sigma: float = 1.3,
                  median: float = 200.0) -> np.ndarray:
    """Log-normal costs: a few arcs dominate the whole tour."""
    return np.exp(rng.normal(np.log(median), sigma, size=(n, n)))


FAMILIES = {
    "uniform": ("uni", None),
    "asymmetric_clustered": ("clu", None),
    "scheduling_constrained": ("sch", None),
    "stacker_crane": ("crn", None),
    "near_symmetric": ("nsy", _near_symmetric),
    "shortest_path_metric": ("met", _shortest_path_metric),
    "heavy_tailed": ("hvy", _heavy_tailed),
}


if HAS_CPLEX:
    class ATSPLazyCallback(LazyConstraintCallback):
        """Lazy constraint callback for dynamic Subtour Elimination Constraints (SEC)."""
        def __init__(self, env):
            super().__init__(env)
            self.n = 0
            self.var_map = {}  # (i, j) -> col_idx

        def __call__(self):
            n = self.n
            sol = {}
            for (i, j), col in self.var_map.items():
                val = self.get_values(col)
                if val > 0.5:
                    sol[i] = j

            # Extract cycles / subtours
            unvisited = set(range(n))
            subtours = []
            while unvisited:
                start = next(iter(unvisited))
                curr = start
                tour = []
                while curr in unvisited:
                    unvisited.remove(curr)
                    tour.append(curr)
                    curr = sol.get(curr, start)
                if curr == start:
                    subtours.append(tour)

            # If subtours exist, add SEC cuts
            if len(subtours) > 1:
                for subtour in subtours:
                    if len(subtour) < n:
                        S = set(subtour)
                        cut_vars = []
                        cut_coefs = []
                        for i in S:
                            for j in S:
                                if i != j and (i, j) in self.var_map:
                                    cut_vars.append(self.var_map[(i, j)])
                                    cut_coefs.append(1.0)
                        self.add(
                            constraint=cplex.SparsePair(ind=cut_vars, val=cut_coefs),
                            sense="L",
                            rhs=float(len(S) - 1)
                        )


_COMMUNITY_WARNING_SHOWN = False

def _solve_exact(dist: np.ndarray, seed: int, time_limit: float = EXACT_TIME_LIMIT,
                 workers: int = WORKERS) -> tuple[float, bool, float]:
    """Return ``(best_cost, proved_optimal, lower_bound)`` for one instance using CPLEX IBM.

    CPLEX solves the ATSP integer program using subtour elimination lazy constraints.
    Returns optimal cost and bound, falling back to local search heuristic if CPLEX limit is reached.
    """
    global _COMMUNITY_WARNING_SHOWN
    n = int(dist.shape[0])
    costs = np.asarray(dist, dtype=np.float64).round().astype(np.int64)

    if not HAS_CPLEX:
        heuristic = reference_cost(dist, effort="medium" if n <= 100 else "low", seed=seed)
        return float(heuristic), False, 0.0

    try:
        prob = cplex.Cplex()
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.set_error_stream(None)

        prob.objective.set_sense(prob.objective.sense.minimize)

        var_map = {}
        col_names = []
        obj_coefs = []
        lb = []
        ub = []
        types = []

        for i in range(n):
            for j in range(n):
                if i != j:
                    col_idx = len(col_names)
                    var_map[(i, j)] = col_idx
                    col_names.append(f"x_{i}_{j}")
                    obj_coefs.append(float(costs[i, j]))
                    lb.append(0.0)
                    ub.append(1.0)
                    types.append("B")

        prob.variables.add(
            obj=obj_coefs,
            lb=lb,
            ub=ub,
            types=types,
            names=col_names
        )

        for i in range(n):
            vars_i = [var_map[(i, j)] for j in range(n) if j != i]
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=vars_i, val=[1.0] * len(vars_i))],
                senses=["E"],
                rhs=[1.0]
            )

        for j in range(n):
            vars_j = [var_map[(i, j)] for i in range(n) if i != j]
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=vars_j, val=[1.0] * len(vars_j))],
                senses=["E"],
                rhs=[1.0]
            )

        cb = prob.register_callback(ATSPLazyCallback)
        cb.n = n
        cb.var_map = var_map

        prob.parameters.timelimit.set(float(time_limit))
        if hasattr(prob.parameters, "threads"):
            prob.parameters.threads.set(int(workers))
        prob.parameters.randomseed.set(int(seed % (2**31)))

        prob.solve()

        status = prob.solution.get_status()
        proved = status in (1, 101, 102)

        try:
            exact = float(prob.solution.get_objective_value())
        except Exception:
            exact = None

        try:
            bound = float(prob.solution.MIP.get_best_objective_value())
        except Exception:
            bound = 0.0

        if proved and exact is not None:
            return exact, True, bound

        heuristic = reference_cost(dist, effort="medium" if n <= 100 else "low", seed=seed)
        best = heuristic if exact is None else min(exact, heuristic)
        return float(best), False, bound

    except Exception as exc:
        err_msg = str(exc)
        if "1016" in err_msg or "Community Edition" in err_msg:
            if not _COMMUNITY_WARNING_SHOWN:
                print("\n  [CPLEX Notice] Capped by CPLEX Community Edition limit (1,000 variables).")
                print("  For n >= 32 with CPLEX exact solver, install full IBM CPLEX Studio via")
                print("  free IBM Academic Initiative: https://www.ibm.com/academic")
                print("  Falling back to reference solver for larger instances...\n")
                _COMMUNITY_WARNING_SHOWN = True
        else:
            print(f"  [CPLEX note: {exc}] Falling back to reference solver for n={n}")
        heuristic = reference_cost(dist, effort="medium" if n <= 100 else "low", seed=seed)
        return float(heuristic), False, 0.0


def generate_atsp_testset_cplex(out_dir: str = OUT_DIR, families: dict = FAMILIES,
                                sizes: tuple = SIZES, seed: int = SEED,
                                exact_time_limit: float = EXACT_TIME_LIMIT,
                                workers: int = WORKERS, force: bool = False) -> str:
    """Write one ``.atsp`` file per (method, size) plus ``bestSolutions.txt`` using CPLEX solver.

    Args:
        out_dir:          destination directory; created if missing.
        families:         ``{name: (short_code, generator or None)}``.
        sizes:            cities per instance — one instance per (method, size).
        seed:             base seed.
        exact_time_limit: seconds CPLEX may spend proving one instance.
        workers:          CPLEX search workers.
        force:            rebuild instances whose file already exists.

    Returns:
        The output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "bestSolutions.txt")

    known = load_best_known(best_path) if os.path.exists(best_path) else {}
    best: dict[str, int] = {}
    proved_count = 0
    started = time.perf_counter()

    total = len(families) * len(sizes)
    print(f"Synthetic ATSP test set (CPLEX IBM) -> {os.path.relpath(out_dir, REPO_ROOT)}")
    print(f"{total} instances ({len(families)} methods x {len(sizes)} sizes, "
          f"n={sizes[0]}..{sizes[-1]}), CPLEX limit {exact_time_limit:.0f}s each\n")

    for n in sizes:
        for family, (code, maker) in families.items():
            name = f"syn_{code}{n}"
            path = os.path.join(out_dir, f"{name}.atsp")

            if os.path.exists(path) and not force and name in known:
                best[name] = int(known[name])
                print(f"  {name:<14} n={n:<5} ref={best[name]:>12,.0f}  "
                      f"(kept, --force to rebuild)")
                continue

            key = f"{seed}|{family}|{n}".encode("utf-8")
            inst_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(),
                                       "big") >> 1

            if maker is None:
                dist = generate_instance(family, n, inst_seed)
            else:
                dist = np.round(maker(n, np.random.default_rng(inst_seed)))
                dist = np.maximum(dist, 1.0)
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
                note = f"reference={ref:.0f} (proven optimal, CPLEX)"
                label = "optimal"
            else:
                slack = (f"<={(ref / bound - 1.0) * 100.0:.2f}% high" if bound > 0
                         else "no lower bound found")
                note = (f"reference={ref:.0f} (upper bound, optimum in "
                        f"[{bound:.0f}, {ref:.0f}], {slack})")
                label = f"bound, {slack}"

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

            _, reparsed = parse_atsp_file(path)
            expected = matrix.astype(np.float64)
            np.fill_diagonal(expected, 0.0)
            if not np.array_equal(reparsed, expected):
                raise AssertionError(f"{name}: file does not round-trip through "
                                     f"parse_atsp_file")

            best[name] = int(round(ref))
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
    generate_atsp_testset_cplex(force="--force" in sys.argv)
