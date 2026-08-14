"""Evaluate an MCTS-AHD-designed `heuristics` desirability matrix for ATSP ACO.

    python -u eval.py <problem_size> <root_dir> <train|val|test>

Prints the objective (mean optimality gap, %) as the last stdout line.

This is upstream MCTS-AHD's `tsp_aco` task, retargeted: the pheromone update
stays fixed Ant System and the LLM designs the *heuristic matrix* — the prior
desirability of each arc that multiplies the pheromone in the transition rule.
Budgets live in `configs/llm/MCTS-AHD/cfg/evaluation/atsp_aco.yaml`.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from aco import ACO  # noqa: E402
from atsp_utils import budget, load_instances, positive_matrix, report  # noqa: E402

try:
    from gpt import heuristics_v2 as heuristics
except Exception:
    from gpt import heuristics

TASK = "atsp_aco"


def solve(instance, settings: dict) -> float:
    # The heuristic sees a matrix with no zeros in it, so `1 / distance_matrix`
    # is finite; see atsp_utils.positive_matrix for why ATSP needs more than
    # upstream's diagonal-only guard. Tour costs use the true matrix.
    heu = heuristics(positive_matrix(instance.dist))

    aco = ACO(instance.dist, heu,
              n_ants=int(settings.get("n_ants", 20)),
              seed=int(settings.get("seed", 2024)))
    time_limit = settings.get("time_limit")
    deadline = None if time_limit is None else time.perf_counter() + float(time_limit)
    return float(aco.run(int(settings.get("n_iterations", 30)), deadline=deadline))


if __name__ == "__main__":
    print("[*] Running ATSP ant colony optimisation ...")

    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mood = sys.argv[3] if len(sys.argv) > 3 else "train"

    settings = budget(TASK, mood)
    instances = load_instances(mood, problem_size if mood == "train" else None)
    costs = [solve(instance, settings) for instance in instances]
    report(instances, costs)
