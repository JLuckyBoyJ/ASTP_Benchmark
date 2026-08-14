import hydra
import logging 
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.utils import print_hyperlink, set_llm_parallelism
from utils.atsp_logging import (
    LLMTracer, load_dotenv, setup_run_logging, write_best_heuristic, write_meta,
    write_summary,
)


ROOT_DIR = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", ".."))

# Secrets live in envs/.env (git-ignored). Loaded before Hydra resolves ${oc.env:...}
_DOTENV = load_dotenv(REPO_ROOT)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../../../configs/llm/HSEvo/cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Seed the evolutionary-loop RNG forParity / reproducibility.
    seed = cfg.get("seed", 2026)
    if seed is not None:
        import numpy as np
        np.random.seed(int(seed))
        logging.info(f"Seed: {seed}")
        
    set_llm_parallelism(cfg.get("n_parallel", 8))
    
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using LLM: {cfg.get('model', 'gpt-4o-mini')}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")
    for _f in _DOTENV:
        logging.info(f"Loaded secrets from {os.path.relpath(_f, REPO_ROOT)}")

    # Per-run tracking
    setup_run_logging(str(workspace_dir))
    write_meta(str(workspace_dir), cfg, REPO_ROOT)
    tracer = LLMTracer(os.path.join(str(workspace_dir), "llm_calls.jsonl")).install()
    logging.info("Tracing every prompt/response -> llm_calls.jsonl")

    if cfg.algorithm == "hsevo":
        from hsevo import HSEvo as LHH
    else:
        raise NotImplementedError(f"Unsupported algorithm: {cfg.algorithm}")

    start_time = time.time()
    lhh = LHH(cfg, ROOT_DIR)
    best_code_overall, best_code_path_overall = lhh.evolve()
    tracer.remove()

    best_path_txt = write_best_heuristic(
        str(workspace_dir), cfg.problem.problem_name, best_code_overall,
        getattr(lhh, "best_obj_overall", None), best_code_path_overall)
        
    write_summary(str(workspace_dir), {
        "framework": "HSEvo",
        "algorithm": cfg.get("algorithm", "hsevo"),
        "problem": cfg.problem.problem_name,
        "problem_type": cfg.problem.problem_type,
        "model": cfg.get("model", "gpt-4o-mini"),
        "objective_name": "mean_gap_percent",
        "best_objective": getattr(lhh, "best_obj_overall", None),
        "function_evals": getattr(lhh, "function_evals", None),
        "minutes": round((time.time() - start_time) / 60.0, 2),
        "best_heuristic_path": best_path_txt,
        "run_dir": str(workspace_dir),
        **tracer.stats(),
    })
    logging.info(f"Best heuristic saved to {print_hyperlink(best_path_txt)}")

    logging.info(f"Best Code Overall: {best_code_overall}")
    if best_code_path_overall:
        best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
        logging.info(f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}")

    # Run validation on test set and write stdout
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w', encoding="utf-8") as file:
        file.writelines((best_code_overall or "") + '\n')
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script...: {print_hyperlink(test_script)}")
    with open(test_script_stdout, 'w', encoding="utf-8") as stdout:
        subprocess.run([sys.executable, test_script, "-1", ROOT_DIR, "test"], stdout=stdout)
    logging.info(f"Validation script finished. Results saved in {print_hyperlink(test_script_stdout)}.")

    with open(test_script_stdout, 'r', encoding="utf-8") as file:
        for line in file.readlines():
            logging.info(line.strip())

if __name__ == "__main__":
    main()
