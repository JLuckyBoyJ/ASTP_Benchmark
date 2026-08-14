"""Bridge between the MCTS-AHD search and one ATSP task.

Two classes, both kept as close to the released MCTS-AHD code as the ATSP
retarget allows:

``Prompts``  assembles the task-specific fragments that every LLM action (i1,
             e1, e2, m1, m2, s1) splices into its prompt: what the problem is,
             what the key heuristic function is called, what it receives and
             what it must return. The *action* prompts themselves — the part
             the paper contributes — live untouched in ``source/evolution.py``.

``Problem``  evaluates a candidate heuristic. MCTS-AHD's contract, inherited
             from ReEvo: write the code to ``problems/<task>/gpt.py``, run
             ``problems/<task>/eval.py`` as a subprocess, read the objective
             off the last line of stdout.

Changes from the released file, all forced and all marked ``ATSP:`` below:

* ``sys.executable`` instead of the string ``"python"`` — macOS and most Linux
  distros ship no bare ``python``, and inside a venv the bare name resolves to
  the wrong interpreter.
* UTF-8 everywhere; the default encoding on Windows mangles LLM output.
* the ``tsp_constructive`` / ``bpp_online`` branches, which imported a
  ``.original.prompts`` package that does not exist in the release, are gone.
* ``inner_run`` could be referenced before assignment when writing ``gpt.py``
  raised; a failed write is now reported as an invalid individual.
* every evaluation appends a line to ``evaluations/index.jsonl`` so a run can
  be audited afterwards without re-reading 300 stdout files.
* an objective of exactly 0 is accepted. Upstream asserts ``obj > 0``; our
  objective is a mean optimality gap in percent, and 0 means the heuristic
  matched the proven optimum on every training instance, which is the best
  possible outcome rather than an error.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any

from utils.utils import block_until_running, file_to_string, filter_traceback

logger = logging.getLogger(__name__)


class Prompts:
    """Task-specific prompt fragments, read from ``prompts/<task>/``."""

    def __init__(self, problem_cfg, root_dir: str):
        self.cfg = problem_cfg
        self.problem = problem_cfg.problem_name
        self.root_dir = root_dir
        self.problem_type = problem_cfg.problem_type
        self.prompt_dir = f"{self.root_dir}/prompts"

        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f"{self.prompt_dir}/{self.problem}{prompt_path_suffix}"

        self.func_signature = file_to_string(
            f"{problem_prompt_path}/func_signature.txt").format(version=2).strip()
        self.func_desc = file_to_string(f"{problem_prompt_path}/func_desc.txt").strip()

        # ATSP: domain hints, the counterpart of ReEvo's `external_knowledge.txt`.
        # They are appended to every action prompt through `get_other_inf()`, so
        # i1/e1/e2/m1/m2/s1 all see them. Disable with
        # `problem.use_external_knowledge=false` to ablate.
        self.external_knowledge = ""
        knowledge_path = f"{problem_prompt_path}/external_knowledge.txt"
        if getattr(problem_cfg, "use_external_knowledge", True) and os.path.isfile(knowledge_path):
            self.external_knowledge = file_to_string(knowledge_path).strip()

        match = re.match(r"^def +(.+?)\((.*)\) *-> *(.*?) *:", self.func_signature)
        assert match is not None, f"unparseable signature: {self.func_signature!r}"
        self.prompt_func_name = match.group(1)
        self.prompt_func_inputs = [txt.split(":")[0].strip()
                                   for txt in match.group(2).split(",")]

        if self.prompt_func_name.startswith("select_next_node"):
            self.prompt_func_outputs = ["next_node"]
        elif self.prompt_func_name.startswith("arc_badness"):
            self.prompt_func_outputs = ["badness"]
        elif self.prompt_func_name.startswith("heuristics"):
            self.prompt_func_outputs = ["heuristics_matrix"]
        elif self.prompt_func_name.startswith("priority"):
            self.prompt_func_outputs = ["priority"]
        else:
            self.prompt_func_outputs = ["result"]

    # -- the interface source/evolution.py expects -----------------------------

    def get_task(self):
        return self.cfg.description

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.func_desc

    def get_other_inf(self):
        if not self.external_knowledge:
            return ""
        return "\nUseful facts about this problem:\n" + self.external_knowledge


class Problem:
    """Evaluate one candidate heuristic for one ATSP task."""

    def __init__(self, cfg, root_dir):
        self.config = cfg
        self.root_dir = root_dir

        self.problem = self.config.problem.problem_name
        self.problem_description = self.config.problem.description
        self.problem_size = self.config.problem.problem_size
        self.obj_type = self.config.problem.obj_type
        self.problem_type = self.config.problem.problem_type
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"
        # ATSP: per-task kill threshold. `cfg.timeout` interpolates
        # `${problem.timeout}`, so each task carries its own.
        self.timeout = int(self.config.get("timeout", 60))

        self.prompts = Prompts(self.config.problem, root_dir)

        # ATSP: eval.py runs as a standalone subprocess with no Hydra context,
        # so the two config choices it needs travel as environment variables.
        # `data` selects configs/llm/MCTS-AHD/cfg/data/<name>.yaml (the split);
        # the budget file is always named after the task.
        self.child_env = dict(os.environ)
        self.child_env["ATSP_DATA"] = str(self.config.get("data", "tsplib"))
        self.child_env["ATSP_EVAL_BUDGET"] = self.problem

        self.iteration = 0
        self.n_evaluated = 0
        self.n_failed = 0
        self.best_obj = float("inf")
        self.eval_dir = os.path.abspath("./evaluations")
        os.makedirs(self.eval_dir, exist_ok=True)
        self.index_path = os.path.join(self.eval_dir, "index.jsonl")

    # -- bookkeeping -----------------------------------------------------------

    def response_to_individual(self, code, response_id, file_name=None) -> dict:
        runid = abs(hash(code))
        file_name = (os.path.join(self.eval_dir, f"problem_eval{runid}.txt")
                     if file_name is None else file_name + ".txt")
        with open(file_name, "w", encoding="utf-8") as fh:
            fh.writelines((code or "") + "\n")

        std_out_filepath = (os.path.join(self.eval_dir, f"problem_eval{runid}_stdout.txt")
                            if file_name is None
                            else file_name[:-4] + "_stdout.txt")
        return {
            "stdout_filepath": std_out_filepath,
            "code_path": os.path.join(self.eval_dir, f"problem_eval{runid}_code.py"),
            "code": code,
            "response_id": response_id,
        }

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def _record(self, individual: dict, seconds: float) -> None:
        """One JSON line per evaluation, for auditing a finished run."""
        self.n_evaluated += 1
        obj = individual.get("obj", float("inf"))
        ok = bool(individual.get("exec_success"))
        if not ok:
            self.n_failed += 1
        elif obj < self.best_obj:
            self.best_obj = obj
        row = {
            "eval": self.n_evaluated,
            "iteration": self.iteration,
            "objective": None if obj == float("inf") else obj,
            "exec_success": ok,
            "seconds": round(seconds, 2),
            "best_so_far": None if self.best_obj == float("inf") else self.best_obj,
            "stdout": os.path.relpath(individual["stdout_filepath"], self.eval_dir),
            "error": (individual.get("traceback_msg") or "").strip().splitlines()[-1:]
                     or None,
        }
        try:
            with open(self.index_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:  # pragma: no cover
            logger.warning("could not append to %s: %s", self.index_path, exc)

    # -- evaluation ------------------------------------------------------------

    def batch_evaluate(self, codes: list[str], iteration: int = 0) -> str | list[Any]:
        """Score every code string; returns the objectives, or ``'timeout'``.

        MCTS-AHD calls this with a single candidate at a time, so evaluation is
        sequential and a wall-clock deadline means the same thing on every call
        — unlike ReEvo, which evaluates a whole generation concurrently.
        """
        self.iteration = iteration
        population = [self.response_to_individual(resp, index)
                      for index, resp in enumerate(codes)]

        for response_id in range(len(population)):
            individual = population[response_id]
            started = time.time()
            inner_run = None

            if individual["code"] is None:
                population[response_id] = self.mark_invalid_individual(
                    individual, "Invalid response!")
                self._record(population[response_id], time.time() - started)
                continue

            try:
                with open(self.output_file, "w", encoding="utf-8") as fh:
                    fh.writelines(individual["code"] + "\n")

                eval_name = "eval.py" if self.problem_type != "black_box" else "eval_black_box.py"
                file_path = f"{self.root_dir}/problems/{self.problem}/{eval_name}"
                with open(individual["stdout_filepath"], "w", encoding="utf-8") as fh:
                    # ATSP: sys.executable, not "python" — there is often no
                    # bare `python`, and inside a venv it is the wrong one.
                    inner_run = subprocess.Popen(
                        [sys.executable, "-u", file_path, f"{self.problem_size}",
                         self.root_dir, "train"],
                        stdout=fh, stderr=fh, env=self.child_env)
                block_until_running(individual["stdout_filepath"], log_status=True,
                                    iter_num=iteration, response_id=response_id)
            except Exception as exc:
                logger.info("Error for response_id %s: %s", response_id, exc)
                population[response_id] = self.mark_invalid_individual(individual, str(exc))
                self._record(population[response_id], time.time() - started)
                continue

            try:
                inner_run.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                logger.info("Timeout for response_id %s: %s", response_id, exc)
                population[response_id] = self.mark_invalid_individual(individual, str(exc))
                inner_run.kill()
                inner_run.communicate()
                self._record(population[response_id], time.time() - started)
                # MCTS-AHD treats a timeout as "abandon this expansion", not as
                # a bad heuristic, so the sentinel propagates up unchanged.
                return "timeout"

            with open(individual["stdout_filepath"], "r", encoding="utf-8",
                      errors="replace") as fh:
                stdout_str = fh.read()
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == "":
                try:
                    individual["obj"] = float(stdout_str.split("\n")[-2])
                    # ATSP: 0 is legal (see the module docstring); NaN is not.
                    assert individual["obj"] == individual["obj"], "objective is NaN"
                    assert individual["obj"] >= 0, "negative gap means a broken reference"
                    individual["obj"] = (-individual["obj"] if self.obj_type == "max"
                                         else individual["obj"])
                    individual["exec_success"] = True
                except Exception:
                    population[response_id] = self.mark_invalid_individual(
                        individual, "Invalid std out / objective value!")
            else:
                population[response_id] = self.mark_invalid_individual(
                    individual, traceback_msg)

            self._record(population[response_id], time.time() - started)
            logger.info("Iteration %s, response_id %s: objective %s",
                        iteration, response_id, population[response_id]["obj"])

        return [indiv["obj"] for indiv in population]
