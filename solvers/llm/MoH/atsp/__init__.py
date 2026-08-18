"""MoH's own ATSP layer: instances, tour primitives, local search.

A deliberate copy rather than a shared package — each solver folder is
self-contained, so MoH can be moved, vendored or diffed on its own, exactly
as EoH, ReEvo, HSEvo and MCTS-AHD each carry their own. The files here are
byte-identical to theirs, which is the point: the frameworks then design
heuristics for the same instances, with the same local search, scored the
same way, so a difference in the reported gap is a difference between the
search methods.

The algorithm drivers MoH actually uses live next to the tasks in
`problems/atsp_*`, because MoH's design space differs: its guided local
search calls the LLM's rule at every perturbation step rather than
precomputing a static guide matrix once per instance.
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
