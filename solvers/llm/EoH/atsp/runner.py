"""Build and run an ATSP EoH experiment from a config file."""

from __future__ import annotations

import glob
import json
import os
import time

from . import _bootstrap  # noqa: F401
from .baselines import baseline_code, baseline_name, load_baseline
from .config import dump_config, load_config, repo_root
from .data import describe_split, resolve_split
from .llm_trace import LLMTracer
from .logging_utils import (
    Stopwatch,
    collect_meta,
    create_run_dir,
    get_logger,
    read_json,
    setup_run_logging,
    write_json,
)
from .registry import TASK_SUMMARY, build_problem, get_problem_class

logger = get_logger()


# ── helpers ───────────────────────────────────────────────────────────────────

def _prepare(config: dict, run_dir: str) -> None:
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(dump_config(config))


def _load_train_instances(config: dict, root: str):
    return resolve_split(config["data"]["train"], root, log=logger.info)


def _describe_instances(instances) -> str:
    sizes = sorted({ins.n for ins in instances})
    kinds = sorted({ins.ref_kind for ins in instances})
    families = sorted({str(ins.meta.get("family", ins.source)) for ins in instances})
    span = f"n={sizes[0]}" if len(sizes) == 1 else f"n={sizes[0]}..{sizes[-1]}"
    return (f"{len(instances)} instances, {span}, families={'/'.join(families)}, "
            f"reference={'/'.join(kinds)}")


def _collect_best(run_dir: str) -> dict | None:
    """Best individual of the run: prefer samples_best, fall back to pops_best."""
    best_path = os.path.join(run_dir, "results", "samples", "samples_best.json")
    best = None
    if os.path.exists(best_path):
        try:
            best = read_json(best_path)
        except (OSError, json.JSONDecodeError):
            best = None

    for path in sorted(glob.glob(os.path.join(run_dir, "results", "pops_best", "*.json"))):
        try:
            candidate = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None
        if not candidate or candidate.get("objective") is None:
            continue
        if best is None or best.get("objective") is None or \
                candidate["objective"] < best["objective"]:
            best = candidate
    return best


def _write_best_heuristic(run_dir: str, task: str, best: dict, fitness_name: str) -> str | None:
    if not best or not best.get("code"):
        return None
    path = os.path.join(run_dir, "best_heuristic.py")
    thought = (best.get("algorithm") or "").strip().replace("\n", " ")
    header = (
        f'"""Best heuristic evolved by EoH for the ATSP `{task}` task.\n\n'
        f"{fitness_name} on the training set: {best.get('objective')}\n"
        f"run: {os.path.relpath(run_dir, repo_root())}\n\n"
        f"Thought: {thought}\n"
        '"""\n\n'
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + best["code"].strip() + "\n")
    return path


# ── smoke test (no LLM) ───────────────────────────────────────────────────────

def smoke(config: dict) -> dict:
    """Evaluate the hand-crafted baseline once — checks data, engine and timing."""
    root = repo_root()
    task = config["task"]["name"]
    run_dir = create_run_dir(os.path.join(root, config["run"]["output_root"]),
                             task, tag=(config["run"].get("tag") or "") + "smoke")
    setup_run_logging(run_dir, debug=config["eoh"]["debug"])
    _prepare(config, run_dir)

    logger.info("=" * 62)
    logger.info("  ATSP EoH smoke test — task '%s' (no LLM calls)", task)
    logger.info("  %s", TASK_SUMMARY[task])
    logger.info("=" * 62)

    instances = _load_train_instances(config, root)
    logger.info("[data] train: %s", _describe_instances(instances))

    problem = build_problem(task, instances, config["task"].get("params"),
                            timeout=config["task"].get("timeout"), n_processes=1)
    logger.info("[task] %s", json.dumps(problem.describe()))

    func = load_baseline(task)
    logger.info("[smoke] evaluating baseline heuristic '%s'", baseline_name(task))

    per_instance = []
    with Stopwatch() as watch:
        for instance in instances:
            t0 = time.time()
            cost = problem.solve_instance(func, instance)
            per_instance.append({
                "instance": instance.name, "n": instance.n, "cost": cost,
                "ref_cost": instance.ref_cost, "gap_percent": instance.gap(cost),
                "seconds": round(time.time() - t0, 2),
            })
            logger.info("  %-24s n=%-4d cost=%-12.2f gap=%7.3f%%  (%.2fs)",
                        instance.name, instance.n, cost,
                        instance.gap(cost), per_instance[-1]["seconds"])
        fitness = problem.evaluate_program(baseline_code(task), func)

    summary = {
        "mode": "smoke",
        "task": task,
        "baseline": baseline_name(task),
        "fitness_name": problem.fitness_name,
        "fitness": fitness,
        "seconds": round(watch.seconds, 1),
        "per_instance": per_instance,
        "run_dir": run_dir,
    }
    write_json(os.path.join(run_dir, "summary.json"), summary)
    logger.info("-" * 62)
    logger.info("  baseline %s = %.4f   (%.1fs total, ~%.1fs per evaluation)",
                problem.fitness_name, fitness, watch.seconds, watch.seconds)
    logger.info("  logs: %s", os.path.relpath(run_dir, root))
    logger.info("-" * 62)
    return summary


# ── full evolution ────────────────────────────────────────────────────────────

def evolve(config: dict) -> dict:
    from eoh import EoH, LLMConfig

    root = repo_root()
    task = config["task"]["name"]
    get_problem_class(task)  # fail fast on a bad task name

    run_dir = create_run_dir(os.path.join(root, config["run"]["output_root"]),
                             task, tag=config["run"].get("tag") or "")
    setup_run_logging(run_dir, debug=config["eoh"]["debug"])
    _prepare(config, run_dir)

    logger.info("=" * 62)
    logger.info("  EoH for ATSP — task '%s'", task)
    logger.info("  %s", TASK_SUMMARY[task])
    logger.info("  run dir: %s", os.path.relpath(run_dir, root))
    logger.info("=" * 62)

    instances = _load_train_instances(config, root)
    logger.info("[data] train: %s", _describe_instances(instances))

    problem = build_problem(task, instances, config["task"].get("params"),
                            timeout=config["task"].get("timeout"),
                            n_processes=config["eoh"]["num_evaluators"])
    logger.info("[task] %s", json.dumps(problem.describe()))

    for dotenv_file in config.get("_dotenv") or []:
        logger.info("[env] loaded %s", os.path.relpath(dotenv_file, root))

    llm_cfg = config["llm"]
    if not llm_cfg.get("use_local") and not llm_cfg.get("api_key"):
        raise SystemExit(
            "No LLM API key. Put it in envs/.env (copy envs/.env.example), export it "
            "(`export OPENAI_API_KEY=sk-...`), or pass --set llm.api_key=...")

    llm = LLMConfig(
        api_endpoint=llm_cfg.get("api_endpoint"),
        api_key=llm_cfg.get("api_key"),
        model=llm_cfg.get("model"),
        use_local=bool(llm_cfg.get("use_local")),
        local_url=llm_cfg.get("local_url"),
        timeout=int(llm_cfg.get("timeout", 150)),
    )

    tracer = None
    if config["run"].get("trace_llm", True):
        tracer = LLMTracer(os.path.join(run_dir, "llm_calls.jsonl")).install()
        logger.info("[trace] every prompt/response -> llm_calls.jsonl")

    meta = collect_meta(root, {
        "task": task,
        "run_dir": run_dir,
        "config": config.get("_config_path"),
        "dotenv": config.get("_dotenv"),
        "model": llm_cfg.get("model"),
        "train": _describe_instances(instances),
        "train_spec": describe_split(config["data"]["train"]),
        "task_settings": problem.describe(),
    })
    write_json(os.path.join(run_dir, "meta.json"), meta)

    ec = config["eoh"]
    eoh = EoH(
        llm=llm,
        problem=problem,
        pop_size=int(ec["pop_size"]),
        n_pop=int(ec["n_pop"]),
        operators=list(ec["operators"]),
        operator_weights=ec.get("operator_weights"),
        n_parents=int(ec["n_parents"]),
        num_samplers=int(ec["num_samplers"]),
        num_evaluators=int(ec["num_evaluators"]),
        max_sample_nums=ec.get("max_sample_nums"),
        output_dir=run_dir,
        debug=bool(ec["debug"]),
        use_seed=bool(ec.get("use_seed")),
        seed_path=ec.get("seed_path") or os.path.join(run_dir, "seeds.json"),
        use_continue=bool(ec.get("use_continue")),
        continue_path=ec.get("continue_path") or os.path.join(
            run_dir, "results", "pops", "population_generation_0.json"),
        continue_id=int(ec.get("continue_id", 0)),
    )

    with Stopwatch() as watch:
        try:
            eoh.run()
        finally:
            if tracer is not None:
                tracer.remove()

    best = _collect_best(run_dir)
    best_path = _write_best_heuristic(run_dir, task, best or {}, problem.fitness_name)

    summary = {
        "mode": "evolve",
        "task": task,
        "model": llm_cfg.get("model"),
        "fitness_name": problem.fitness_name,
        "best_objective": (best or {}).get("objective"),
        "best_thought": (best or {}).get("algorithm"),
        "best_heuristic_path": best_path,
        "minutes": round(watch.seconds / 60.0, 2),
        "run_dir": run_dir,
        "config": config.get("_config_path"),
        "train": _describe_instances(instances),
        "train_spec": describe_split(config["data"]["train"]),
        "task_settings": problem.describe(),
        "eoh": {k: ec[k] for k in ("pop_size", "n_pop", "n_parents", "operators",
                                   "num_samplers", "num_evaluators", "max_sample_nums")},
    }
    if tracer is not None:
        summary.update(tracer.stats())
    write_json(os.path.join(run_dir, "summary.json"), summary)

    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["minutes"] = summary["minutes"]
    write_json(os.path.join(run_dir, "meta.json"), meta)

    logger.info("-" * 62)
    logger.info("  best %s = %s", problem.fitness_name, summary["best_objective"])
    if best_path:
        logger.info("  best heuristic: %s", os.path.relpath(best_path, root))
    logger.info("  run directory : %s", os.path.relpath(run_dir, root))
    if tracer is not None:
        logger.info("  LLM calls     : %d (%d failed, %.0fs total)",
                    tracer.n_calls, tracer.n_failures, tracer.total_seconds)
    logger.info("")
    logger.info("  Benchmark it on the held-out TSPLIB ATSP set:")
    logger.info("    python python_scripts/llm/EoH/eval_eoh_atsp.py --run %s",
                os.path.relpath(run_dir, root))
    logger.info("-" * 62)
    return summary


def run_from_cli(config_path: str, overrides: list[str] | None = None,
                 smoke_only: bool = False) -> dict:
    config = load_config(config_path, overrides)
    return smoke(config) if smoke_only else evolve(config)
