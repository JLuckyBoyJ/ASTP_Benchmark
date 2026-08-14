"""The Monte Carlo tree itself — the paper's Section 3.2 and 3.3, unchanged.

Every node other than the virtual root holds one executable heuristic and its
linguistic description. ``Q`` is the best (negated) objective anywhere in the
node's subtree; ``visits`` is the MCTS visit count; ``subtree`` is the list of
nodes below the depth-1 ancestor, which action e1 samples from.

Three fixes relative to the released file, all marked ``ATSP:``:

* ``__repr__`` referenced ``self.answer``, which no node has, so printing or
  logging any node raised ``AttributeError``.
* ``uct`` divided by ``q_max - q_min`` with no guard. The two are equal
  whenever every node evaluated so far shares one objective — which happens on
  the very first selection if initialisation produced a single distinct value —
  and the run died with ``ZeroDivisionError``. The paper's normalisation is
  kept; only the degenerate case is handled, using the ``epsilon`` the class
  already declared.
* the exploration constant, the widening exponent and the depth cap are taken
  from the config instead of being hardcoded, so Table 5's ablations can be run
  from the command line.
"""

from __future__ import annotations

import math


class MCTSNode:
    def __init__(self, algorithm, code, obj, depth=0, is_root=False, parent=None,
                 visit=0, raw_info=None, Q=0, action=None):
        self.algorithm = algorithm
        self.code = code
        self.parent = parent
        self.depth = depth
        self.children = []
        self.children_info = []
        self.visits = visit
        self.subtree = []
        self.raw_info = raw_info
        self.Q = Q
        self.reward = -1 * obj
        # ATSP: which LLM action created this node (i1/e1/e2/m1/m2/s1), so the
        # serialised tree says where each heuristic came from.
        self.action = action

    def add_child(self, child_node: "MCTSNode"):
        self.children.append(child_node)

    def __repr__(self):  # ATSP: was `self.answer`, which does not exist
        return (f"MCTSNode(depth={self.depth}, action={self.action}, "
                f"Q={self.Q:.4f}, visits={self.visits}, "
                f"children={len(self.children)})")


class MCTS:
    def __init__(self, root_answer, exploration_constant_0: float = 0.1,
                 alpha: float = 0.5, max_depth: int = 10,
                 discount_factor: float = 1.0):
        self.exploration_constant_0 = exploration_constant_0   # lambda_0, Eq. (7)
        self.alpha = alpha                                     # progressive widening
        self.max_depth = max_depth
        self.epsilon = 1e-10
        self.discount_factor = discount_factor                 # constant 1 in the paper
        self.q_min = 0
        self.q_max = -10000
        self.rank_list = []

        self.root = MCTSNode(algorithm=root_answer, code=root_answer, depth=0,
                             obj=0, is_root=True, action="root")

        # Logs
        self.critiques = []
        self.refinements = []
        self.rewards = []
        self.selected_nodes = []

    def backpropagate(self, node: MCTSNode):
        """Eq. (6): a parent's Q is the best Q among its children."""
        if node.Q not in self.rank_list:
            self.rank_list.append(node.Q)
            self.rank_list.sort()
        self.q_min = min(self.q_min, node.Q)
        self.q_max = max(self.q_max, node.Q)
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = parent.Q * (1 - self.discount_factor) + best_child_Q * self.discount_factor
            parent.visits += 1
            if parent.code != 'Root' and parent.parent.code == 'Root':
                parent.subtree.append(node)
            parent = parent.parent

    def uct(self, node: MCTSNode, eval_remain):
        """Eq. (5) with the exploration decay of Eq. (7)."""
        self.exploration_constant = self.exploration_constant_0 * eval_remain
        # ATSP: guard the degenerate range instead of dividing by zero.
        span = self.q_max - self.q_min
        normalised = 0.0 if abs(span) < self.epsilon else (node.Q - self.q_min) / span
        return normalised + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / max(node.visits, 1)
        )

    def is_fully_expanded(self, node: MCTSNode):
        return (len(node.children) >= self.max_children
                or any(child.Q > node.Q for child in node.children)
                or node.code == 'Root')
