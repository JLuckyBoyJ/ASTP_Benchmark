"""Evaluate a MoH-designed `select_next_node` for greedy ATSP construction.

    python -u eval.py <problem_size> <root_dir> <train|val|test> [timeout]

Prints the utility (mean optimality gap, %) as the last stdout line.

`<problem_size>` is the subtask size; MoH trains one meta-optimizer against
several sizes at once and weights their utilities by size (Eq. 2), so each
evaluation is scoped to instances of exactly that size.

The signature follows the constructive task on the MCTS-AHD and ReEvo sides
verbatim — `unvisited_nodes` is a **set**, not an array — so a heuristic written
by any of the four frameworks is structurally comparable. The only difference
from their symmetric-TSP originals is that the matrix is directed.

There is no engine budget here: a tour is built in one pass and that is the
whole evaluation, which is why `cfg/evaluation/atsp_constructive.yaml` is empty
rather than absent. The task is configured like the others instead of being a
special case.
"""

import os
import sys
from copy import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from atsp_utils import load_instances, report  # noqa: E402

from gpt import select_next_node  # noqa: E402

TASK = "atsp_constructive"


def solve(instance) -> float:
    """Build one tour, calling the heuristic once per step."""
    n = instance.n
    dist = instance.dist
    start_node = 0
    solution = [start_node]
    unvisited = set(range(n))
    unvisited.remove(start_node)

    for _ in range(n - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist.copy(),
        )
        next_node = int(next_node)
        if next_node not in unvisited:
            raise ValueError(f"select_next_node returned an invalid node: {next_node}")
        solution.append(next_node)
        unvisited.remove(next_node)

    tour = np.asarray(solution, dtype=np.int64)
    return float(dist[tour, np.roll(tour, -1)].sum())


if __name__ == "__main__":
    print("[*] Running ATSP greedy construction ...")

    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mood = sys.argv[3] if len(sys.argv) > 3 else "val"

    instances = load_instances(mood, problem_size)
    costs = [solve(instance) for instance in instances]
    report(instances, costs)
