"""MCTS-AHD on ATSP — entry point.

    cd solvers/llm/MCTS-AHD
    python main.py problem=atsp_gls

Hydra reads its configuration from ``configs/llm/MCTS-AHD/cfg`` (outside the
solver, with the rest of the repository's configs) and writes each run to
``runs/llm/MCTS-AHD/<problem>-<type>/<date>_<time>/``.

The run directory is self-contained:

    run.log            everything logged *and* printed, including the UCT
                       rank list and the action taken at each expansion
    llm_calls.jsonl    every prompt and response, tagged with its MCTS action
    meta.json          config snapshot, git commit, platform, argv
    mcts_tree.json     the search tree: objective, Q, visits, depth per node
    population/        the elite set after each expansion round
    evaluations/       stdout of every heuristic evaluated + index.jsonl
    best_heuristic.py  the winner, runnable
    summary.json       best objective, LLM statistics, wall time
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra

from utils.utils import init_client, print_hyperlink
from utils.atsp_logging import (
    LLMTracer, load_dotenv, setup_run_logging, tee_stdout, write_best_heuristic,
    write_mcts_tree, write_meta, write_summary,
)

ROOT_DIR = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", ".."))

# Secrets live in envs/.env (git-ignored). Loaded before Hydra resolves
# ${oc.env:OPENAI_API_KEY}; a real environment variable still takes precedence.
_DOTENV = load_dotenv(REPO_ROOT)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None,
            config_path="../../../configs/llm/MCTS-AHD/cfg",
            config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Solver root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using LLM: {cfg.llm_client.model}")
    logging.info(f"Using algorithm: {cfg.algorithm}")
    logging.info(f"Budget: max_fe={cfg.max_fe}, N_I={cfg.init_pop_size}, "
                 f"|E|={cfg.pop_size}, lambda_0={cfg.exploration_constant}, "
                 f"alpha={cfg.progressive_widening_alpha}")
    for _f in _DOTENV:
        logging.info(f"Loaded secrets from {os.path.relpath(_f, REPO_ROOT)}")

    setup_run_logging(str(workspace_dir))
    restore_streams = tee_stdout(str(workspace_dir))
    write_meta(str(workspace_dir), cfg, REPO_ROOT)
    tracer = LLMTracer(os.path.join(str(workspace_dir), "llm_calls.jsonl")).install()
    logging.info("Tracing every prompt/response -> llm_calls.jsonl")

    client = init_client(cfg)

    if cfg.algorithm != "mcts_ahd":
        raise NotImplementedError(
            f"this solver implements 'mcts_ahd', not {cfg.algorithm!r}; "
            "EoH and ReEvo live under solvers/llm/EoH and solvers/llm/ReEvo")

    from ahd_adapter import AHD as LHH

    lhh = LHH(cfg, ROOT_DIR, workspace_dir, client)
    start_time = time.time()
    try:
        best_code_overall, best_code_path_overall = lhh.evolve()
    finally:
        tracer.remove()

    best_obj = getattr(lhh, "best_obj_overall", None)
    best_path_txt = write_best_heuristic(
        str(workspace_dir), cfg.problem.problem_name, best_code_overall,
        best_obj, best_code_path_overall)

    tree_path = None
    if getattr(lhh, "mcts_root", None) is not None:
        tree_path = write_mcts_tree(str(workspace_dir), lhh.mcts_root, {
            "problem": cfg.problem.problem_name,
            "max_fe": cfg.max_fe,
            "function_evals": getattr(lhh, "function_evals", None),
            "best_objective": best_obj,
        })
        logging.info(f"MCTS tree saved to {print_hyperlink(tree_path)}")

    write_summary(str(workspace_dir), {
        "framework": "MCTS-AHD",
        "algorithm": cfg.get("algorithm", "mcts_ahd"),
        "problem": cfg.problem.problem_name,
        "problem_type": cfg.problem.problem_type,
        "model": cfg.llm_client.model,
        "objective_name": "mean_gap_percent",
        "best_objective": best_obj,
        "function_evals": getattr(lhh, "function_evals", None),
        "n_tree_nodes": getattr(lhh, "n_tree_nodes", None),
        "max_tree_depth": getattr(lhh, "max_tree_depth", None),
        "minutes": round((time.time() - start_time) / 60.0, 2),
        "best_heuristic_path": best_path_txt,
        "mcts_tree_path": tree_path,
        "run_dir": str(workspace_dir),
        **tracer.stats(),
    })
    logging.info(f"Best heuristic saved to {print_hyperlink(best_path_txt)}")
    logging.info(f"Best objective (training split): {best_obj}")

    # Quick sanity check on the small end of the benchmark. The full sweep is
    # python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py.
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py",
              "w", encoding="utf-8") as fh:
        fh.writelines(best_code_overall + "\n")
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script: {print_hyperlink(test_script)}")
    with open(test_script_stdout, "w", encoding="utf-8") as stdout:
        subprocess.run([sys.executable, test_script, "-1", ROOT_DIR, "test"],
                       stdout=stdout, stderr=subprocess.STDOUT)
    with open(test_script_stdout, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh.readlines():
            logging.info(line.strip())

    restore_streams()


if __name__ == "__main__":
    main()
