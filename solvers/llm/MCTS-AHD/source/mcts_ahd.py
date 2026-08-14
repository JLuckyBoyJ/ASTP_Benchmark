"""MCTS-AHD: Monte Carlo tree search over LLM-designed heuristics.

This is the paper's method, kept intact:

* ``N_I`` initial heuristics are generated (action i1 for the first, action e1
  for the rest) and attached to a virtual root;
* each round runs *selection* (UCT with the normalised Q of Eq. 5 and the
  exploration decay of Eq. 7), *expansion* (2k+2 children via actions m1, m2,
  e2, s1), *simulation* (evaluate the new heuristics on the training split) and
  *backpropagation* (Eq. 6: a parent's Q is the best Q in its subtree);
* progressive widening adds a new child to an already-visited node when
  ``floor(N(n)^alpha) >= |children(n)|``, using e1 at the root and e2 elsewhere;
* nothing is ever discarded. A heuristic that scores badly stays in the tree
  and can still be developed later, which is the paper's central claim.

Additions for this repository, all marked ``ATSP:`` and none of them touching
the search itself: the parameters come from the config, each node records the
action that produced it, and the run appends a line to ``progress.jsonl`` and
refreshes ``mcts_tree.json`` after every round so an interrupted run is still
readable.
"""

import copy
import json
import os
import random
import time

from .mcts import MCTS, MCTSNode
from .evolution_interface import InterfaceEC


class MCTS_AHD:
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage

        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # Experimental settings
        self.init_size = paras.init_size          # N_I
        self.pop_size = paras.pop_size            # |E|, the elite set
        self.fe_max = paras.ec_fe_max             # T
        self.eval_times = 0

        self.operators = paras.ec_operators
        self.operator_weights = list(paras.ec_operator_weights)
        # ATSP: k from the config rather than a hardcoded 2. The expansion
        # produces k children with m1, k with m2 and one each with e2 and s1.
        k = int(paras.mcts_expansion_children)
        for index, name in enumerate(self.operators):
            if name in ("m1", "m2"):
                self.operator_weights[index] = k
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode
        self.ndelay = 1

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        # ATSP: MCTS knobs, from the config
        self.exploration_constant = paras.mcts_exploration_constant
        self.alpha = paras.mcts_alpha
        self.max_depth = paras.mcts_max_depth
        self.seed = paras.mcts_seed

        # ATSP: tracking
        self.run_dir = os.path.abspath(os.path.join(self.output_path, os.pardir))
        self.progress_path = os.path.join(self.run_dir, "progress.jsonl")
        self.tree_path = os.path.join(self.run_dir, "mcts_tree.json")
        self.started_at = time.time()
        self.mcts_root = None
        self.best_objective = None
        self.interface_ec = None

        print("- MCTS-AHD parameters loaded -")
        random.seed(self.seed)

    # ── tracking helpers (ATSP) ───────────────────────────────────────────────

    def _log_progress(self, action: str, parent_obj, obj, depth: int,
                      accepted: bool, note: str = "") -> None:
        row = {
            "eval": self.eval_times,
            "max_fe": self.fe_max,
            "action": action,
            "parent_objective": None if parent_obj in (None, float("inf")) else parent_obj,
            "objective": None if obj in (None, float("inf")) else obj,
            "depth": depth,
            "accepted": accepted,
            "best_so_far": self.best_objective,
            "elapsed_s": round(time.time() - self.started_at, 1),
        }
        if note:
            row["note"] = note
        try:
            with open(self.progress_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except OSError:  # pragma: no cover
            pass

    def n_tree_nodes(self) -> int:
        if self.mcts_root is None:
            return 0

        def count(node):
            return 1 + sum(count(child) for child in node.children)
        return count(self.mcts_root) - 1          # the virtual root is not a heuristic

    def tree_depth(self) -> int:
        if self.mcts_root is None:
            return 0

        def depth(node):
            return 1 + max((depth(child) for child in node.children), default=0)
        return depth(self.mcts_root) - 1

    def _dump_tree(self) -> None:
        """Refresh mcts_tree.json so an interrupted run is still inspectable."""
        if self.mcts_root is None:
            return
        try:
            from utils.atsp_logging import write_mcts_tree
            write_mcts_tree(self.run_dir, self.mcts_root, {
                "function_evals": self.eval_times,
                "max_fe": self.fe_max,
                "best_objective": self.best_objective,
                "n_nodes": self.n_tree_nodes(),
                "depth": self.tree_depth(),
            })
        except Exception:  # pragma: no cover - tracking must never kill a run
            pass

    def _note_objective(self, obj) -> None:
        if obj is None or obj == float("inf"):
            return
        if self.best_objective is None or obj < self.best_objective:
            self.best_objective = float(obj)

    # ── population ────────────────────────────────────────────────────────────

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['algorithm'] == offspring['algorithm']:
                if self.debug_mode:
                    print("duplicated result, retrying ... ")
        population.append(offspring)

    # ── expansion ─────────────────────────────────────────────────────────────

    def expand(self, mcts, cur_node, nodes_set, option):
        if option == 's1':
            # Tree-path reasoning: collect the unique heuristics on the path
            # from this leaf up to the root and ask for one that improves on
            # all of them (paper, action s1).
            path_set = []
            now = copy.deepcopy(cur_node)
            while now.code != "Root":
                path_set.append(now.raw_info)
                now = copy.deepcopy(now.parent)
            path_set = self.manage.population_management_s1(path_set, len(path_set))
            if len(path_set) == 1:
                return nodes_set
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                self.eval_times, path_set, cur_node.raw_info, cur_node.children_info, option)
        elif option == 'e1':
            # Root-level crossover: one heuristic sampled from each depth-1
            # subtree, so the new node diverges from all of them.
            e1_set = [copy.deepcopy(children.subtree[random.choices(
                range(len(children.subtree)), k=1)[0]].raw_info)
                for children in mcts.root.children]
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                self.eval_times, e1_set, cur_node.raw_info, cur_node.children_info, option)
        else:
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                self.eval_times, nodes_set, cur_node.raw_info, cur_node.children_info, option)

        if offsprings is None:
            print(f"Timeout emerge, no expanding with action {option}.")
            self._log_progress(option, self._obj_of(cur_node), None,
                               cur_node.depth + 1, False, "timeout or no valid offspring")
            return nodes_set

        self._note_objective(offsprings['objective'])

        if option != 'e1':
            print(f"Action: {option}, Father Obj: {cur_node.raw_info['objective']}, "
                  f"Now Obj: {offsprings['objective']}, Depth: {cur_node.depth + 1}")
        else:
            if self.interface_ec.check_duplicate_obj(mcts.root.children_info,
                                                     offsprings['objective']):
                print(f"Duplicated e1, no action, Father is Root, "
                      f"Abandon Obj: {offsprings['objective']}")
                self._log_progress(option, None, offsprings['objective'], 1, False,
                                   "duplicate objective at root")
                return nodes_set
            print(f"Action: {option}, Father is Root, Now Obj: {offsprings['objective']}")

        if offsprings['objective'] != float('inf'):
            self.add2pop(nodes_set, offsprings)
            size_act = min(len(nodes_set), self.pop_size)
            nodes_set = self.manage.population_management(nodes_set, size_act)
            nownode = MCTSNode(offsprings['algorithm'], offsprings['code'],
                               offsprings['objective'], parent=cur_node,
                               depth=cur_node.depth + 1, visit=1,
                               Q=-1 * offsprings['objective'], raw_info=offsprings,
                               action=option)
            if option == 'e1':
                nownode.subtree.append(nownode)
            cur_node.add_child(nownode)
            cur_node.children_info.append(offsprings)
            mcts.backpropagate(nownode)
            self._log_progress(option, self._obj_of(cur_node), offsprings['objective'],
                               nownode.depth, True)
        else:
            self._log_progress(option, self._obj_of(cur_node), None,
                               cur_node.depth + 1, False, "heuristic failed to run")
        return nodes_set

    @staticmethod
    def _obj_of(node):
        if isinstance(node.raw_info, dict):
            return node.raw_info.get("objective")
        return None

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("- Initialization start -")

        interface_prob = self.prob
        self.interface_ec = InterfaceEC(
            self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode,
            interface_prob, use_local_llm=self.use_local_llm, url=self.url,
            select=self.select, n_p=self.exp_n_proc, timeout=self.timeout,
            use_numba=self.use_numba)

        brothers = []
        mcts = MCTS('Root', exploration_constant_0=self.exploration_constant,
                    alpha=self.alpha, max_depth=self.max_depth)
        self.mcts_root = mcts.root
        n_op = len(self.operators)

        # N_I initial nodes: the first from scratch (i1), the rest diverging
        # from what already exists (e1).
        for index in range(self.init_size):
            action = "i1" if index == 0 else "e1"
            self.eval_times, brothers, offsprings = self.interface_ec.get_algorithm(
                self.eval_times, brothers, action)
            if offsprings is None:
                raise RuntimeError(
                    f"initialisation failed: no valid heuristic after {self.eval_times} "
                    f"evaluations. Check evaluations/index.jsonl in this run directory "
                    f"— every candidate raised, timed out, or printed no objective.")
            brothers.append(offsprings)
            self._note_objective(offsprings['objective'])
            nownode = MCTSNode(offsprings['algorithm'], offsprings['code'],
                               offsprings['objective'], parent=mcts.root, depth=1,
                               visit=1, Q=-1 * offsprings['objective'],
                               raw_info=offsprings, action=action)
            mcts.root.add_child(nownode)
            mcts.root.children_info.append(offsprings)
            mcts.backpropagate(nownode)
            nownode.subtree.append(nownode)
            self._log_progress(action, None, offsprings['objective'], 1, True)
            print(f"Initial node {index + 1}/{self.init_size} "
                  f"({action}): objective {offsprings['objective']}")

        nodes_set = brothers
        nodes_set = self.manage.population_management(
            nodes_set, min(len(nodes_set), self.pop_size))
        self._dump_tree()
        print("- Initialization finished - evolution start -")

        while self.eval_times < self.fe_max:
            print(f"Current performances of MCTS nodes: {mcts.rank_list}")
            cur_node = mcts.root

            # Selection, with progressive widening on the way down.
            while len(cur_node.children) > 0 and cur_node.depth < mcts.max_depth:
                uct_scores = [mcts.uct(node, max(1 - self.eval_times / self.fe_max, 0))
                              for node in cur_node.children]
                selected_pair_idx = uct_scores.index(max(uct_scores))
                if int(cur_node.visits ** mcts.alpha) > len(cur_node.children):
                    op = 'e1' if cur_node == mcts.root else self.operators[1]  # e2
                    nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                cur_node = cur_node.children[selected_pair_idx]

            # Expansion + simulation: 2k+2 children of the selected leaf.
            for i in range(n_op):
                op = self.operators[i]
                op_w = self.operator_weights[i]
                if op_w:
                    print(f"Iter: {self.eval_times}/{self.fe_max} OP: {op}", end="|")
                for _ in range(op_w):
                    nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                assert len(cur_node.children) == len(cur_node.children_info)

            filename = os.path.join(
                self.output_path, f"population_generation_{self.eval_times}.json")
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(nodes_set, f, indent=5, default=str)

            best_filename = os.path.join(
                self.output_path, f"best_population_generation_{self.eval_times}.json")
            with open(best_filename, 'w', encoding="utf-8") as f:
                json.dump(nodes_set[0], f, indent=5, default=str)

            self._dump_tree()

        self.best_objective = nodes_set[0]["objective"]
        best_filename = os.path.join(self.output_path, "best_final.json")
        with open(best_filename, 'w', encoding="utf-8") as f:
            json.dump(nodes_set[0], f, indent=5, default=str)
        self._dump_tree()
        return nodes_set[0]["code"], best_filename
