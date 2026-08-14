"""Evaluate an MCTS-AHD-designed `heuristics` guide matrix for ATSP guided local search.

    python -u eval.py <problem_size> <root_dir> <train|val|test>

Prints the objective (mean optimality gap, %) as the last stdout line.

Everything that used to be a constant at the top of this file now lives in
`configs/llm/MCTS-AHD/cfg/evaluation/atsp_gls.yaml`, and the instances in
`cfg/data/<ATSP_DATA>.yaml`. The reasoning behind those numbers is in the YAML,
next to the numbers themselves; the short version is that training is bounded
by iterations rather than wall clock so a candidate's score cannot depend on
machine load, and the iteration count is chosen to spend the same effective
budget the benchmark does.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from atsp_utils import budget, load_instances, report, resolve_iter_limit  # noqa: E402
from gls import guided_local_search  # noqa: E402

try:
    from gpt import heuristics_v2 as heuristics
except Exception:
    from gpt import heuristics

TASK = "atsp_gls"


def solve(instance, settings: dict) -> float:
    guide = heuristics(instance.dist.copy())
    _, cost = guided_local_search(
        instance.dist, guide,
        perturbation_moves=int(settings.get("perturbation_moves", 30)),
        iter_limit=resolve_iter_limit(settings.get("iter_limit", 1000), instance.n),
        time_limit=settings.get("time_limit"),
    )
    return cost


if __name__ == "__main__":
    print("[*] Running ATSP guided local search ...")

    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mood = sys.argv[3] if len(sys.argv) > 3 else "train"

    settings = budget(TASK, mood)
    instances = load_instances(mood, problem_size if mood == "train" else None)
    costs = [solve(instance, settings) for instance in instances]
    report(instances, costs)
