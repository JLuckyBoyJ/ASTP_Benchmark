import hydra
import logging 
import os
from pathlib import Path
import subprocess
import sys
import time
from utils.utils import init_client, print_hyperlink
from utils.atsp_logging import (
    LLMTracer, load_dotenv, setup_run_logging, write_best_heuristic, write_meta,
    write_summary,
)


ROOT_DIR = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", ".."))

# Secrets live in envs/.env (git-ignored). Loaded before Hydra resolves
# ${oc.env:OPENAI_API_KEY}; a real environment variable still takes precedence.
_DOTENV = load_dotenv(REPO_ROOT)
logging.basicConfig(level=logging.INFO)

# Configs live with the rest of the repo's configs, not inside the solver:
#   configs/llm/ReEvo/cfg/{config.yaml, problem/, llm_client/, hydra/}
@hydra.main(version_base=None,
            config_path="../../../configs/llm/ReEvo/cfg",
            config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")
    for _f in _DOTENV:
        logging.info(f"Loaded secrets from {os.path.relpath(_f, REPO_ROOT)}")

    # Per-run tracking: provenance up front, every LLM prompt/response as it
    # happens, a summary at the end. See utils/atsp_logging.py.
    setup_run_logging(str(workspace_dir))
    write_meta(str(workspace_dir), cfg, REPO_ROOT)
    tracer = LLMTracer(os.path.join(str(workspace_dir), "llm_calls.jsonl")).install()
    logging.info("Tracing every prompt/response -> llm_calls.jsonl")

    client = init_client(cfg)
    # optional clients for operators (ReEvo)
    long_ref_llm = hydra.utils.instantiate(cfg.llm_long_ref) if cfg.get("llm_long_ref") else None
    short_ref_llm = hydra.utils.instantiate(cfg.llm_short_ref) if cfg.get("llm_short_ref") else None
    crossover_llm = hydra.utils.instantiate(cfg.llm_crossover) if cfg.get("llm_crossover") else None
    mutation_llm = hydra.utils.instantiate(cfg.llm_mutation) if cfg.get("llm_mutation") else None
    
    if cfg.algorithm == "reevo":
        from reevo import ReEvo as LHH
    elif cfg.algorithm == "ael":
        from baselines.ael.ga import AEL as LHH
    elif cfg.algorithm == "eoh":
        from baselines.eoh import EoH as LHH
    else:
        raise NotImplementedError

    # Main algorithm
    if cfg.algorithm != "reevo":
        lhh = LHH(cfg, ROOT_DIR, client)
    else:
        lhh = LHH(cfg, ROOT_DIR, client, long_reflector_llm=long_ref_llm, short_reflector_llm=short_ref_llm, 
                  crossover_llm=crossover_llm, mutation_llm=mutation_llm)
        
    start_time = time.time()
    best_code_overall, best_code_path_overall = lhh.evolve()
    tracer.remove()

    best_path_txt = write_best_heuristic(
        str(workspace_dir), cfg.problem.problem_name, best_code_overall,
        getattr(lhh, "best_obj_overall", None), best_code_path_overall)
    write_summary(str(workspace_dir), {
        "framework": "ReEvo",
        "algorithm": cfg.get("algorithm", "reevo"),
        "problem": cfg.problem.problem_name,
        "problem_type": cfg.problem.problem_type,
        "model": cfg.get("model", None) or cfg.llm_client.model,
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
    best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
    logging.info(f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}")
    
    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w', encoding="utf-8") as file:
        file.writelines(best_code_overall + '\n')
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script...: {print_hyperlink(test_script)}")
    with open(test_script_stdout, 'w', encoding="utf-8") as stdout:
        subprocess.run([sys.executable, test_script, "-1", ROOT_DIR, "test"], stdout=stdout)
    logging.info(f"Validation script finished. Results are saved in {print_hyperlink(test_script_stdout)}.")
    
    # Print the results
    with open(test_script_stdout, 'r', encoding="utf-8") as file:
        for line in file.readlines():
            logging.info(line.strip())

if __name__ == "__main__":
    main()
