"""Adapter between the Hydra config and the MCTS-AHD search object.

Upstream's version of this file constructs a ``Paras`` bag with the paper's
defaults hardcoded. Here every one of them comes from
``configs/llm/MCTS-AHD/cfg/config.yaml``, so the paper's settings are visible,
overridable from the command line and recorded in each run's ``meta.json``:

    N_I         init_pop_size            paper default 4
    |E|         pop_size                 paper default 10
    T           max_fe                   paper default 1000
    lambda_0    exploration_constant     paper default 0.1
    alpha       progressive_widening_alpha  paper default 0.5
    k           expansion_children       paper default 2  (2k+2 children/expansion)

After ``evolve()`` returns, the adapter exposes what ``main.py`` needs for the
run summary: ``best_obj_overall``, ``function_evals``, ``mcts_root`` (for the
tree dump) and the tree's size and depth.
"""

from __future__ import annotations

import os

from source.mcts_ahd import MCTS_AHD
from source.getParas import Paras
from source import prob_rank, pop_greedy
from problem_adapter import Problem


class AHD:
    def __init__(self, cfg, root_dir, workdir, client) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.workdir = str(workdir)
        self.problem = Problem(cfg, root_dir)

        # The elite set and the per-expansion population snapshots go in their
        # own folder; upstream drops one JSON per expansion into the run root.
        population_dir = os.path.join(self.workdir, "population")
        os.makedirs(population_dir, exist_ok=True)

        self.paras = Paras()
        self.paras.set_paras(
            method="mcts_ahd",
            init_size=cfg.init_pop_size,
            pop_size=cfg.pop_size,
            llm_model=client,
            ec_fe_max=cfg.max_fe,
            ec_m=cfg.get("crossover_parents", 5),
            exp_output_path=population_dir + os.sep,
            exp_debug_mode=bool(cfg.get("debug", False)),
            eva_timeout=cfg.timeout,
            # MCTS parameters from the paper (Sections 3.2 and 3.3)
            mcts_exploration_constant=cfg.get("exploration_constant", 0.1),
            mcts_alpha=cfg.get("progressive_widening_alpha", 0.5),
            mcts_max_depth=cfg.get("max_tree_depth", 10),
            mcts_expansion_children=cfg.get("expansion_children", 2),
            mcts_seed=cfg.get("seed", 2024),
        )

        self.best_obj_overall = None
        self.function_evals = 0
        self.mcts_root = None
        self.n_tree_nodes = 0
        self.max_tree_depth = 0

    def evolve(self):
        print("- Evolution start -")

        method = MCTS_AHD(self.paras, self.problem, prob_rank, pop_greedy)
        code, path = method.run()

        self.best_obj_overall = method.best_objective
        self.function_evals = method.eval_times
        self.mcts_root = method.mcts_root
        self.n_tree_nodes = method.n_tree_nodes()
        self.max_tree_depth = method.tree_depth()

        print("> End of evolution")
        print("-----------------------------------------")
        print("---  MCTS-AHD successfully finished!  ---")
        print("-----------------------------------------")
        return code, path
