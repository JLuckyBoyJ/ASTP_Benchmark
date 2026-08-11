"""Evaluate a ReEvo-designed `heuristics` desirability matrix for ATSP ACO.

    python -u eval.py <problem_size> <root_dir> <train|val|test>

Prints the objective (mean optimality gap, %) as the last stdout line.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time  # noqa: E402

import numpy as np  # noqa: E402

from aco import ACO  # noqa: E402
from atsp_utils import load_instances, report  # noqa: E402

try:
    from gpt import heuristics_v2 as heuristics
except Exception:
    from gpt import heuristics

N_ITERATIONS = 30
N_ANTS = 20
TIME_LIMIT = 10.0
VAL_ITERATIONS = 50
VAL_TIME_LIMIT = 30.0


def solve(instance, n_iterations: int, time_limit: float) -> float:
    # Upstream sets the diagonal to a non-zero value before calling the
    # heuristic so that rules like 1/d do not divide by zero; self-loops
    # are excluded during construction regardless.
    dist = instance.dist.copy()
    np.fill_diagonal(dist, 1.0)
    heu = heuristics(dist)
    aco = ACO(instance.dist, heu, n_ants=N_ANTS, seed=2024)
    return aco.run(n_iterations, deadline=time.perf_counter() + time_limit)


if __name__ == "__main__":
    print("[*] Running ATSP ant colony optimisation ...")

    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mood = sys.argv[3] if len(sys.argv) > 3 else "train"

    instances = load_instances(mood, problem_size if mood == "train" else None)
    iterations = N_ITERATIONS if mood == "train" else VAL_ITERATIONS
    budget = TIME_LIMIT if mood == "train" else VAL_TIME_LIMIT
    costs = [solve(instance, iterations, budget) for instance in instances]
    report(instances, costs)
