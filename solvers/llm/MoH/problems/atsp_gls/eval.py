"""Evaluate a MoH-designed `update_edge_distance` rule for ATSP guided local search.

    python -u eval.py <problem_size> <root_dir> <train|val|test> [timeout]

Prints the utility (mean optimality gap, %) as the last stdout line, which is
what `moh.py: evaluate_heuristic` parses.

`<problem_size>` is the **subtask size**. MoH is multi-task: one run trains a
single meta-optimizer against `atsp_gls-50`, `atsp_gls-100`, ... at once, and
each of those subtasks is scored only on instances of its own size, because the
meta-utility in Eq. (2) weights them by size. `atsp_utils.load_instances` raises
rather than falling back if a size has no instances — see the note there.

The engine is the same guided local search the ReEvo, MCTS-AHD and HSEvo sides
use (`problems/atsp_gls/gls.py`), and the budget comes from
`configs/llm/MoH/cfg/evaluation/atsp_gls.yaml`. Neither is duplicated here.

MoH's design space, versus the other frameworks'
------------------------------------------------
ReEvo and MCTS-AHD ask for a **static** guide matrix computed once from the
distance matrix. MoH's own TSP-GLS task asks for something strictly more
general: a function that is called *every perturbation step* and may re-weight
arcs using the current local optimum and the penalty history. That is MoH's
formulation and it is kept here — it is the harder design space, and the
paper's improvement-heuristic results are all in it.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from atsp_utils import budget, load_instances, report, resolve_iter_limit  # noqa: E402
from gls import guided_local_search  # noqa: E402

from gpt import update_edge_distance  # noqa: E402

from concurrent.futures import ProcessPoolExecutor

TASK = "atsp_gls"


def solve(instance, settings: dict) -> float:
    _, cost = guided_local_search(
        instance.dist, update_edge_distance,
        perturbation_moves=int(settings.get("perturbation_moves", 30)),
        iter_limit=resolve_iter_limit(settings.get("iter_limit", 1000), instance.n),
        time_limit=settings.get("time_limit"),
    )
    return cost


def _solve_worker(args):
    instance, settings = args
    return solve(instance, settings)


if __name__ == "__main__":
    print("[*] Running ATSP guided local search ...")

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

