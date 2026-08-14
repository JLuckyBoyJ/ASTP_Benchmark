"""Evaluate a ReEvo-designed `select_next_node` for greedy ATSP construction.

    python -u eval.py <problem_size> <root_dir> <train|val|test>

Prints the objective (mean optimality gap, %) as the last stdout line.

The signature follows ReEvo's `tsp_constructive`: `unvisited_nodes` is a
**set**, not an array. Kept as upstream so a heuristic written for their TSP
task is structurally comparable to one written here.
"""

import os
import sys
from copy import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from atsp_utils import load_instances, report  # noqa: E402

try:
    from gpt import select_next_node_v2 as select_next_node
except Exception:
    from gpt import select_next_node


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
    mood = sys.argv[3] if len(sys.argv) > 3 else "train"

    instances = load_instances(mood, problem_size if mood == "train" else None)
    costs = [solve(instance) for instance in instances]
    report(instances, costs)
