"""RunLogger: the centralised I/O manager for a single MoH run.

Everything a run writes goes through here, so the layout of a run directory is
defined in one place:

    logs/utility.csv           one row per subtask utility measured
    logs/meta_utility.csv      one row per outer-loop iteration
    progress.jsonl             the same, machine-readable, appended live so an
                               interrupted run can still be read
    pop/improver/<tag>.json    the optimizer population P after each iteration
    pop/subtask/<tag>.json     the heuristic populations H_i after each iteration
    code/improver/iter_N.py    the accepted optimizer at iteration N
    code/improver/candidate_*  every optimizer proposed, accepted or not
    evaluations/<name>.txt     stdout of every heuristic evaluation
    evaluations/index.jsonl    one line per evaluation: subtask, mode, utility

The candidate optimizers are kept even when rejected, deliberately: Appendix E
of the paper is a gallery of the strategies MoH invents (ACO-like, PSO-like,
simulated annealing, tabu, hybrids), and the rejected ones are the evidence for
the claim that the outer loop explores rather than drifting.

``evaluations/`` replaces upstream's habit of dropping evaluation stdout into
the same folder as the logs; the naming and the index match the MCTS-AHD and
ReEvo run directories, so ``python_scripts/llm/MoH/list_runs.py`` and its
siblings read all four frameworks the same way.
"""

import csv
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class RunLogger:
    """Centralised I/O manager for a single MoH run."""

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Hydra output directory. Defaults to cwd (Hydra sets cwd).
        """
        self.output_dir = output_dir or os.getcwd()

        self.dirs = {
            "pop_improver":  os.path.join(self.output_dir, "pop", "improver"),
            "pop_subtask":   os.path.join(self.output_dir, "pop", "subtask"),
            "code_improver": os.path.join(self.output_dir, "code", "improver"),
            "evaluations":   os.path.join(self.output_dir, "evaluations"),
            "logs":          os.path.join(self.output_dir, "logs"),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self._init_csv(self._utility_csv_path(),
                       ["iteration", "subtask", "utility", "timestamp"])
        self._init_csv(self._meta_csv_path(),
                       ["iteration", "meta_utility", "accepted", "eval_calls",
                        "timestamp"])
        self._candidate_counter = 0
        self._eval_counter = 0

    # ── paths ────────────────────────────────────────────────────────────────

    def _utility_csv_path(self):
        return os.path.join(self.dirs["logs"], "utility.csv")

    def _meta_csv_path(self):
        return os.path.join(self.dirs["logs"], "meta_utility.csv")

    def _progress_path(self):
        return os.path.join(self.output_dir, "progress.jsonl")

    def _eval_index_path(self):
        return os.path.join(self.dirs["evaluations"], "index.jsonl")

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _init_csv(path, headers):
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)

    @staticmethod
    def _now():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _append_jsonl(path, record):
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:  # pragma: no cover
            logger.warning("could not append to %s: %s", path, exc)

    # ── evaluations ──────────────────────────────────────────────────────────

    def evaluation_path(self, iteration: int, subtask: str, mode: str) -> str:
        """Reserve a stdout file for one heuristic evaluation."""
        self._eval_counter += 1
        safe = subtask.replace("/", "_")
        name = f"eval_{self._eval_counter:04d}_iter{iteration}_{safe}_{mode}.txt"
        return os.path.join(self.dirs["evaluations"], name)

    def log_evaluation(self, iteration: int, subtask: str, mode: str,
                       utility, path: str, seconds: float = None,
                       status: str = "ok", error: str = None):
        """Append one line to evaluations/index.jsonl."""
        self._append_jsonl(self._eval_index_path(), {
            "eval": self._eval_counter,
            "iteration": iteration,
            "subtask": subtask,
            "mode": mode,
            "utility": utility,
            "seconds": None if seconds is None else round(seconds, 2),
            "status": status,
            "error": error,
            "stdout": os.path.relpath(path, self.output_dir),
            "timestamp": self._now(),
        })

    # ── population snapshots ─────────────────────────────────────────────────

    def save_improver_pop(self, pop, tag: str):
        path = os.path.join(self.dirs["pop_improver"], f"{tag}.json")
        pop.save_all_data_to_file(path)
        logger.debug(f"Saved optimizer population -> {path}")

    def save_subtask_pop(self, pop, tag: str):
        path = os.path.join(self.dirs["pop_subtask"], f"{tag}.json")
        pop.save_all_data_to_file(path)
        logger.debug(f"Saved heuristic populations -> {path}")

    # ── evolved code ─────────────────────────────────────────────────────────

    def save_improver_code(self, code_str: str, iteration: int, idea: str = ""):
        """Save the accepted improve_algorithm for an iteration."""
        path = os.path.join(self.dirs["code_improver"], f"iter_{iteration}.py")
        header = f"# Iteration {iteration}\n# Idea: {idea}\n" if idea else ""
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + code_str + "\n")
        logger.info(f"Saved optimizer code -> {path}")
        return path

    def save_candidate_improver(self, code_str: str, idea: str = "",
                                utility: float = None, note: str = ""):
        """Save a candidate optimizer proposed during meta_utility evaluation.

        Numbered rather than timestamped: two candidates generated inside the
        same second used to overwrite each other, which is exactly the case
        worth keeping — a batch of siblings from one outer-loop invocation.
        """
        self._candidate_counter += 1
        path = os.path.join(self.dirs["code_improver"],
                            f"candidate_{self._candidate_counter:04d}.py")
        lines = []
        if idea:
            lines.append(f"# Idea: {idea}")
        if utility is not None:
            lines.append(f"# Utility: {utility}")
        if note:
            lines.append(f"# Note: {note}")
        lines.append(code_str or "")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    # ── structured logging ───────────────────────────────────────────────────

    def log_utility(self, iteration: int, subtask: str, utility: float):
        with open(self._utility_csv_path(), "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([iteration, subtask, utility, self._now()])

    def log_meta_utility(self, iteration: int, utility: float,
                         accepted: bool = False, eval_calls: int = None):
        with open(self._meta_csv_path(), "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([iteration, utility, accepted, eval_calls,
                                    self._now()])

    def log_progress(self, record: dict):
        """One line per outer iteration, written live.

        ``list_runs.py`` reads the last line of this file to show how far an
        unfinished run got, which is the difference between "it crashed" and
        "somebody pressed Ctrl-C".
        """
        self._append_jsonl(self._progress_path(), {"t": self._now(), **record})

    # ── convenience ──────────────────────────────────────────────────────────

    def save_iteration(self, iteration: int, improver_pop, subtask_pop,
                       improver_code: str = None, improver_idea: str = ""):
        """One call to snapshot everything at the end of an iteration."""
        tag = f"iter_{iteration}"
        self.save_improver_pop(improver_pop, tag)
        self.save_subtask_pop(subtask_pop, tag)
        if improver_code:
            self.save_improver_code(improver_code, iteration, improver_idea)
