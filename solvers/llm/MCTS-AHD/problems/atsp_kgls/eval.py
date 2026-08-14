"""Evaluate an MCTS-AHD-designed `arc_badness` rule for ATSP knowledge-guided local search.

    python -u eval.py <problem_size> <root_dir> <train|val|test>

Prints the objective (mean optimality gap, %) as the last stdout line.

Budgets come from `configs/llm/MCTS-AHD/cfg/evaluation/atsp_kgls.yaml`, which is
deliberately identical to `atsp_gls.yaml`: the two improvement tasks must differ
only in what the LLM designs — a static guide matrix versus a dynamic
arc-badness rule — never in how long the engine runs.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from atsp_utils import budget, load_instances, report, resolve_iter_limit  # noqa: E402
from kgls import knowledge_guided_local_search  # noqa: E402

try:
    from gpt import arc_badness_v2 as arc_badness
except Exception:
    from gpt import arc_badness

TASK = "atsp_kgls"


def solve(instance, settings: dict) -> float:
    _, cost = knowledge_guided_local_search(
        instance.dist, arc_badness,
        perturbation_moves=int(settings.get("perturbation_moves", 30)),
        iter_limit=resolve_iter_limit(settings.get("iter_limit", 1000), instance.n),
        time_limit=settings.get("time_limit"),
    )
    return cost


if __name__ == "__main__":
    print("[*] Running ATSP knowledge-guided local search ...")

    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mood = sys.argv[3] if len(sys.argv) > 3 else "train"

    settings = budget(TASK, mood)
    instances = load_instances(mood, problem_size if mood == "train" else None)
    costs = [solve(instance, settings) for instance in instances]
    report(instances, costs)
