"""Evaluate a ReEvo-designed `heuristics` guide matrix for ATSP guided local search.

Called by ReEvo as::

    python -u eval.py <problem_size> <root_dir> <train|val|test>

and must print the objective (mean optimality gap, %) as the last stdout line.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from atsp_utils import load_instances, report, train_iter_limit  # noqa: E402
from gls import guided_local_search  # noqa: E402

try:
    from gpt import heuristics_v2 as heuristics
except Exception:
    from gpt import heuristics

# Training is bounded by ITERATIONS, not wall clock, and that is deliberate.
# ReEvo evaluates a whole generation as concurrent subprocesses, so under a
# wall-clock budget a candidate's score depends on how many rivals happen to
# share the CPU with it. Measured on this task: the seed scored -2.74 when it
# was evaluated alone at iteration 0 and -1.23 when the identical code was
# re-sampled later alongside nine others — a swing larger than the gap between
# the seed and the best evolved candidate, so the search was ranking machine
# load rather than heuristics, and the elitist never moved in 106 evaluations.
#
# Nothing is lost by dropping the clock here: the guide matrix is computed once
# per instance, so there is no per-iteration cost for a wall-clock budget to
# price. (EoH is the opposite case — its rule is called every iteration, so its
# wall-clock budget is doing real work and stays.)
# The iteration count is chosen per instance size to spend the same effective
# budget the benchmark does (see atsp_utils.train_iter_limit). Matching matters:
# a guide evolved under a 50-iteration search beat the baseline by 0.62 points
# on held-out TSPLIB *at that budget*, and lost to it at 10 s.
PERTURBATION_MOVES = 30

# Benchmarking is a different matter: it runs on its own, and the 10 s budget
# is what the EoH side reports against, so the two stay comparable.
VAL_ITER_LIMIT = 1000
VAL_TIME_LIMIT = 10.0


def solve(instance, iter_limit: int, time_limit: float | None) -> float:
    guide = heuristics(instance.dist.copy())
    _, cost = guided_local_search(
        instance.dist, guide,
        perturbation_moves=PERTURBATION_MOVES,
        iter_limit=iter_limit,
        time_limit=time_limit,
    )
    return cost


if __name__ == "__main__":
    print("[*] Running ATSP guided local search ...")

    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mood = sys.argv[3] if len(sys.argv) > 3 else "train"

    instances = load_instances(mood, problem_size if mood == "train" else None)
    if mood == "train":
        costs = [solve(i, train_iter_limit(i.n), None) for i in instances]
    else:
        costs = [solve(i, VAL_ITER_LIMIT, VAL_TIME_LIMIT) for i in instances]
    report(instances, costs)
