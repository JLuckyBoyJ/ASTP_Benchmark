"""Evaluate a ReEvo-designed `heuristics` guide matrix for ATSP guided local search.

Called by ReEvo as::

    python -u eval.py <problem_size> <root_dir> <train|val|test>

and must print the objective (mean optimality gap, %) as the last stdout line.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from atsp_utils import load_instances, report  # noqa: E402
from gls import guided_local_search  # noqa: E402

try:
    from gpt import heuristics_v2 as heuristics
except Exception:
    from gpt import heuristics

# Budget per instance. Wall-clock bounded on purpose: an O(n^2) guide is free
# under an iteration cap at n=50 and ruinous at n=443, and the search should
# feel the cost of what it designs.
PERTURBATION_MOVES = 30
ITER_LIMIT = 1000
TIME_LIMIT = 5.0
VAL_TIME_LIMIT = 10.0


def solve(instance, time_limit: float) -> float:
    guide = heuristics(instance.dist.copy())
    _, cost = guided_local_search(
        instance.dist, guide,
        perturbation_moves=PERTURBATION_MOVES,
        iter_limit=ITER_LIMIT,
        time_limit=time_limit,
    )
    return cost


if __name__ == "__main__":
    print("[*] Running ATSP guided local search ...")

    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mood = sys.argv[3] if len(sys.argv) > 3 else "train"

    instances = load_instances(mood, problem_size if mood == "train" else None)
    budget = TIME_LIMIT if mood == "train" else VAL_TIME_LIMIT
    costs = [solve(instance, budget) for instance in instances]
    report(instances, costs)
