"""MoH on ATSP — entry point.

    cd solvers/llm/MoH
    python main.py problem=atsp_gls

Hydra reads its configuration from ``configs/llm/MoH/cfg`` (outside the solver,
with the rest of the repository's configs) and writes each run to
``runs/llm/MoH/<problem>-<type>/<date>_<time>/``.

The run directory is self-contained:

    run.log                 everything logged *and* printed
    llm_calls.jsonl         every prompt and response, tagged with the loop it
                            came from (meta = outer, heu = inner) and what it
                            was asking for
    meta.json               config snapshot, git commit, platform, argv
    progress.jsonl          one line per outer iteration, written live
    logs/meta_utility.csv   U(I) per iteration, and whether it was accepted
    logs/utility.csv        U_i(h) per subtask, per iteration
    pop/                    both populations after every iteration
    code/improver/          every candidate optimizer, accepted or not
    evaluations/            stdout of every heuristic evaluation + index.jsonl
    best_meta_optimizer.py  the discovered I*_T — the artefact of the method
    best_heuristic_*.py     the winning heuristic for each subtask
    best_heuristic.py       the winner on the largest subtask, where the shared
                            benchmark scripts look for it
    summary.json            best utilities, LLM statistics, wall time

Two modes. ``mode=train`` (default) runs Algorithm 1: the outer loop designs
heuristic-optimizers, the inner loop applies them to every subtask.
``mode=inference meta_optimizer=<path>`` skips the outer loop and applies a
previously trained optimizer to whatever subtasks the problem config names —
the paper's inference stage, and the way to point an optimizer trained at
n=50/100 at a larger instance size.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra

ROOT_DIR = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from utils.atsp_logging import (  # noqa: E402
    LLMTracer, load_dotenv, setup_run_logging, tee_stdout, write_best_heuristic,
    write_best_meta_optimizer, write_meta, write_summary,
)
from utils.utils import print_hyperlink  # noqa: E402

# Secrets live in envs/.env (git-ignored). Loaded before Hydra resolves
# ${oc.env:OPENAI_API_KEY}; a real environment variable still takes precedence.
_DOTENV = load_dotenv(REPO_ROOT)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None,
            config_path="../../../configs/llm/MoH/cfg",
            config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()          # Hydra has chdir'd into the run directory
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Solver root: {print_hyperlink(ROOT_DIR)}")
    for _f in _DOTENV:
        logging.info(f"Loaded secrets from {os.path.relpath(_f, REPO_ROOT)}")

    setup_run_logging(str(workspace_dir), debug=bool(cfg.get("debug", False)))
    restore_streams = tee_stdout(str(workspace_dir))
    tracer = LLMTracer(os.path.join(str(workspace_dir), "llm_calls.jsonl")).install()
    logging.info("Tracing every prompt/response -> llm_calls.jsonl")

    # Two clients, one per level of the bilevel framework (Section 3.2). They
    # may be different models; `role` is what splits llm_calls.jsonl by loop.
    heu_llm = hydra.utils.instantiate(cfg.heu)
    meta_llm = hydra.utils.instantiate(cfg.meta)
    heu_llm.role, meta_llm.role = "heu", "meta"
    logging.info(f"Inner loop (heuristic design) LLM: {cfg.heu.model}")
    logging.info(f"Outer loop (optimizer design)  LLM: {cfg.meta.model}")

    from moh import MoH
    from utils.run_logger import RunLogger

    run_logger = RunLogger(str(workspace_dir))
    optimizer = MoH(cfg, ROOT_DIR, heu_llm=heu_llm, meta_llm=meta_llm,
                    run_logger=run_logger)

    logging.info(f"Mode: {cfg.get('mode', 'train')} | subtasks: "
                 f"{', '.join(optimizer.subtask_list)}")
    logging.info(f"Budget: n_iterations={cfg.n_iterations}, pop_size={cfg.pop_size}, "
                 f"max_eval_calls={cfg.max_eval_calls} per subtask "
                 f"({optimizer.max_eval_calls or 'unlimited'} total)")
    logging.info(f"Data config: {cfg.data.name} | eval timeout: {cfg.timeout}s")

    write_meta(str(workspace_dir), cfg, REPO_ROOT, optimizer.subtask_list, _DOTENV)

    started = time.time()
    try:
        optimizer.run()
    finally:
        tracer.remove()
        results = optimizer.results()
        _write_artifacts(cfg, workspace_dir, results, tracer, started)

    _validate_headline(cfg, workspace_dir, results)
    logging.info(f"Run directory: {print_hyperlink(workspace_dir)}")
    restore_streams()


def _write_artifacts(cfg, workspace_dir, results, tracer, started):
    """Save the winners and the summary. Runs even if the search raised."""
    best_optimizer = results.get("best_meta_optimizer")
    optimizer_path = None
    if best_optimizer:
        optimizer_path = write_best_meta_optimizer(
            str(workspace_dir), best_optimizer["best_sol"],
            best_optimizer["utility"], best_optimizer.get("iteration"))
        logging.info(f"Best meta-optimizer -> {print_hyperlink(optimizer_path)}")

    heuristic_paths = {}
    for subtask, entry in results.get("best_per_subtask", {}).items():
        heuristic_paths[subtask] = write_best_heuristic(
            str(workspace_dir), cfg.problem.problem_name, subtask,
            entry["best_sol"], entry["utility"])

    # The shared benchmark scripts look for `best_heuristic.py`, so the winner
    # on the largest subtask is written there too — largest because that is the
    # end MoH's generalisation claim is about and the end Eq. (2) weights most.
    headline = results.get("headline_heuristic")
    headline_path = None
    if headline:
        headline_path = write_best_heuristic(
            str(workspace_dir), cfg.problem.problem_name,
            results.get("headline_subtask", "?"), headline["best_sol"],
            headline["utility"], filename="best_heuristic.py")
        logging.info(f"Headline heuristic ({results.get('headline_subtask')}) -> "
                     f"{print_hyperlink(headline_path)}")

    write_summary(str(workspace_dir), {
        "framework": "MoH",
        "algorithm": cfg.get("algorithm", "moh"),
        "mode": cfg.get("mode", "train"),
        "problem": cfg.problem.problem_name,
        "problem_type": cfg.problem.problem_type,
        "subtasks": results.get("subtasks"),
        "heu_model": cfg.heu.model,
        "meta_model": cfg.meta.model,
        "data_config": str(cfg.data.name),
        "objective_name": "mean_gap_percent",
        # The number the outer loop minimises: Eq. (2), size-weighted across
        # subtasks. Not comparable with a single-task framework's objective;
        # compare `best_objective` (below) or the post-hoc TSPLIB evaluation.
        "best_meta_utility": results.get("meta_utility"),
        "best_objective": (results.get("headline_heuristic") or {}).get("utility"),
        "headline_subtask": results.get("headline_subtask"),
        "utility_per_subtask": {s: v["utility"]
                                for s, v in results.get("best_per_subtask", {}).items()},
        "eval_calls": results.get("eval_calls"),
        "eval_budget": results.get("eval_budget"),
        "n_iterations": cfg.n_iterations,
        "pop_size": cfg.pop_size,
        "minutes": round((time.time() - started) / 60.0, 2),
        "best_meta_optimizer_path": optimizer_path,
        "best_heuristic_path": headline_path,
        "best_heuristic_paths": heuristic_paths,
        "run_dir": str(workspace_dir),
        **tracer.stats(),
    })


def _validate_headline(cfg, workspace_dir, results):
    """Sanity-check the winner on the held-out split, in-process-free.

    A quick look only — the full sweep, with per-instance tables and the result
    files the cross-framework report reads, is
    ``python_scripts/llm/MoH/eval_moh_atsp.py``.
    """
    headline = results.get("headline_heuristic")
    if not headline:
        logging.warning("no heuristic survived the run; nothing to validate")
        return
    task = cfg.problem.problem_name
    with open(os.path.join(ROOT_DIR, "problems", task, "gpt.py"), "w",
              encoding="utf-8") as fh:
        fh.write(headline["best_sol"] + "\n")

    env = dict(os.environ)
    env["ATSP_DATA"] = str(cfg.data.name)
    env["ATSP_EVAL_BUDGET"] = str(cfg.get("evaluation_budget", None) or task)
    script = os.path.join(ROOT_DIR, "problems", task, "eval.py")
    stdout_path = os.path.join(str(workspace_dir), "best_heuristic_test_stdout.txt")
    logging.info(f"Validating on the held-out split: {print_hyperlink(script)}")
    with open(stdout_path, "w", encoding="utf-8") as stdout:
        subprocess.run([sys.executable, "-u", script, "0", ROOT_DIR, "test"],
                       stdout=stdout, stderr=subprocess.STDOUT, cwd=ROOT_DIR, env=env)
    with open(stdout_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh.read().splitlines()[-8:]:
            logging.info(line.strip())
    logging.info("Full sweep: python python_scripts/llm/MoH/eval_moh_atsp.py "
                 f"--run {os.path.relpath(str(workspace_dir), REPO_ROOT)}")


if __name__ == "__main__":
    main()
