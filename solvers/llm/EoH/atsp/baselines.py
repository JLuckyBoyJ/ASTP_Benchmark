"""Human-designed reference heuristics, one per task.

They serve three purposes:

1. **Smoke tests** — ``--smoke`` runs a task end-to-end without touching an LLM.
2. **Baselines** — the benchmark tables compare every evolved heuristic against
   the corresponding hand-crafted rule (nearest neighbour, classic GLS penalty,
   Ant System, random ruin).
3. **Optional seeds** — they can be fed to EoH via ``use_seed`` if you want to
   reproduce the paper's "with expert heuristic" ablation.

Each baseline is stored as source code so that the evaluator can load an
evolved heuristic and a baseline through exactly the same code path.
"""

from __future__ import annotations

import types

BASELINES: dict[str, dict[str, str]] = {
    "construct": {
        "name": "nearest_neighbour",
        "entry_point": "select_next_node",
        "code": '''
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Nearest neighbour on outgoing arcs."""
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])])
''',
    },
    "gls": {
        "name": "classic_gls_penalty",
        "entry_point": "update_edge_distance",
        "code": '''
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Voudouris & Tsang guided local search penalty, applied per directed arc.

    Utility of an arc is cost / (1 + times_penalised); the arcs of the current
    local optimum are augmented by that amount so long, repeatedly-used arcs
    become unattractive.
    """
    updated = edge_distance.copy()
    n = len(local_opt_tour)
    for k in range(n):
        u = int(local_opt_tour[k])
        v = int(local_opt_tour[(k + 1) % n])
        updated[u, v] += edge_distance[u, v] / (1.0 + edge_n_used[u, v])
    return updated
''',
    },
    "aco": {
        "name": "ant_system",
        "entry_point": "update_pheromone",
        "code": '''
import numpy as np


def update_pheromone(pheromone, ant_tours, tour_costs, best_tour, best_cost,
                     rho, iteration, max_iterations):
    """Plain Ant System, deposits on traversed arcs only (directed)."""
    n = pheromone.shape[0]
    pheromone = (1.0 - rho) * pheromone
    for tour, cost in zip(ant_tours, tour_costs):
        deposit = 1.0 / max(cost, 1e-12)
        for i in range(n):
            u = int(tour[i])
            v = int(tour[(i + 1) % n])
            pheromone[u, v] += deposit
    return pheromone
''',
    },
    "rnr": {
        "name": "random_ruin",
        "entry_point": "destroy_nodes",
        "code": '''
import numpy as np


def destroy_nodes(current_tour, distance_matrix, n_destroy):
    """Uniform random removal."""
    return np.random.choice(current_tour, size=n_destroy, replace=False)
''',
    },
}


def baseline_code(task: str) -> str:
    try:
        return BASELINES[task]["code"]
    except KeyError:
        raise ValueError(f"No baseline defined for task {task!r}") from None


def baseline_name(task: str) -> str:
    return BASELINES[task]["name"]


def load_callable(code: str, entry_point: str):
    """Compile a heuristic source string and return its entry-point callable."""
    module = types.ModuleType("atsp_heuristic")
    exec(compile(code, "<heuristic>", "exec"), module.__dict__)
    func = getattr(module, entry_point, None)
    if func is None:
        raise ValueError(f"heuristic source does not define {entry_point!r}")
    return func


def load_baseline(task: str):
    spec = BASELINES[task]
    return load_callable(spec["code"], spec["entry_point"])
