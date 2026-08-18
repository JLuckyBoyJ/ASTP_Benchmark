"""Evaluate a MoH-designed `arc_badness` rule for ATSP knowledge-guided local search.

    python -u eval.py <problem_size> <root_dir> <train|val|test> [timeout]

Prints the utility (mean optimality gap, %) as the last stdout line.

`<problem_size>` is the subtask size; each subtask is scored only on instances
of its own size, because MoH weights subtask utilities by size (Eq. 2).

The engine (`kgls.py`) and the budget
(`configs/llm/MoH/cfg/evaluation/atsp_kgls.yaml`) are shared, byte for byte,
with the MCTS-AHD side. That file is in turn deliberately identical to
`atsp_gls.yaml`: the two improvement tasks must differ only in *what the LLM
designs* — a per-iteration cost matrix versus a per-arc badness vector — never
in how long the engine is allowed to run.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from atsp_utils import budget, load_instances, report, resolve_iter_limit  # noqa: E402
from kgls import knowledge_guided_local_search  # noqa: E402

from gpt import arc_badness  # noqa: E402

from concurrent.futures import ProcessPoolExecutor

TASK = "atsp_kgls"


def solve(instance, settings: dict) -> float:
    _, cost = knowledge_guided_local_search(
        instance.dist, arc_badness,
        perturbation_moves=int(settings.get("perturbation_moves", 30)),
        iter_limit=resolve_iter_limit(settings.get("iter_limit", 1000), instance.n),
        time_limit=settings.get("time_limit"),
    )
    return cost


def _solve_worker(args):
    instance, settings = args
    return solve(instance, settings)


if __name__ == "__main__":
    print("[*] Running ATSP knowledge-guided local search ...")

    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mood = sys.argv[3] if len(sys.argv) > 3 else "val"

    settings = budget(TASK, mood)
    instances = load_instances(mood, problem_size)
    if len(instances) > 1:
        max_workers = min(len(instances), os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            costs = list(executor.map(_solve_worker, [(inst, settings) for inst in instances]))
    else:
        costs = [solve(instance, settings) for instance in instances]
    report(instances, costs)

