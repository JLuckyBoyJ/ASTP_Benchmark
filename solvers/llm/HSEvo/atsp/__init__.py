"""ReEvo's own ATSP layer: instances, tour primitives, local search.

A deliberate copy rather than a shared package — each solver folder is
self-contained, so ReEvo can be moved, vendored or diffed on its own. The
algorithm drivers ReEvo actually uses (guided local search with a static
guide matrix, ACO with a directed pheromone update) live next to the tasks
in `problems/atsp_*`, because they differ from EoH's.
"""

from .data import ATSPInstance, load_tsplib_atsp, resolve_split
from .engines import (
    candidate_lists,
    is_valid_tour,
    local_search,
    local_search_around,
    nearest_neighbour_tour,
    tour_cost,
)

__all__ = ["ATSPInstance", "load_tsplib_atsp", "resolve_split", "tour_cost",
           "is_valid_tour", "nearest_neighbour_tour", "candidate_lists",
           "local_search", "local_search_around"]
