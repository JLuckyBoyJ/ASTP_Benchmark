#!/usr/bin/env python
"""One line per MoH run — the index into runs/llm/MoH/.

Every run writes its own `run.log`, `summary.json` and `progress.jsonl`, which
is what you want when inspecting *one* run and useless when you have thirty.
This walks them all and prints a table, so you can see at a glance which run was
best, which ones failed, and what each cost.

    python python_scripts/llm/MoH/list_runs.py
    python python_scripts/llm/MoH/list_runs.py --task atsp_gls
    python python_scripts/llm/MoH/list_runs.py --sort test --csv runs/index.csv
    python python_scripts/llm/MoH/list_runs.py --unfinished     # what died

Two objectives are shown because MoH minimises two different things:

`meta`  U(I) from Eq. (2) — the size-weighted mean utility of the best
        heuristic-optimizer. It is what the outer loop selects on, and it is not
        comparable with a single-task framework's number.
`train` the utility of the headline heuristic on the largest subtask, which is
        comparable with the other frameworks' training objective.
`test`  the mean gap on the held-out TSPLIB set, once the run has been scored by
        `eval_moh_atsp.py`. This is the number to report.

A run counts as finished when it has `summary.json`; an unfinished one is shown
with whatever `progress.jsonl` got to before it stopped, which is usually enough
to tell a crash from a Ctrl-C.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "llm", "MoH")


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _last_progress(run_dir: str) -> dict:
    path = os.path.join(run_dir, "progress.jsonl")
    last = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = json.loads(line)
    except Exception:
        pass
    return last


def collect(task: str | None = None) -> list[dict]:
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(RUNS, "*", "*", "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        meta = _read_json(meta_path)
        if task and meta.get("problem") != task:
            continue
        summary = _read_json(os.path.join(run_dir, "summary.json"))
        progress = _last_progress(run_dir)
        evaluation = _read_json(os.path.join(run_dir, "eval_test.json"))

        finished = bool(summary)
        rows.append({
            "run": os.path.relpath(run_dir, RUNS),
            "task": meta.get("problem", "?"),
            "mode": meta.get("mode", "train"),
            "model": meta.get("heu_model", "?"),
            "data": meta.get("data_config", "?"),
            "subtasks": ",".join(str(s).rsplit("-", 1)[-1]
                                 for s in (meta.get("subtasks") or [])),
            "started": meta.get("started_at", ""),
            "status": "ok" if finished else "unfinished",
            "meta_utility": (summary.get("best_meta_utility")
                             if finished else progress.get("meta_utility")),
            "train_objective": (summary.get("best_objective")
                                if finished else None),
            "test_gap": (evaluation.get("summary", {}) or {}).get("mean_gap_percent"),
            "iters": (summary.get("n_iterations")
                      if finished else progress.get("iteration")),
            "evals": (summary.get("eval_calls")
                      if finished else progress.get("eval_calls")),
            "minutes": summary.get("minutes"),
            "llm_calls": summary.get("llm_calls"),
            "run_dir": run_dir,
        })
    return rows


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return "-"
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def table(rows: list[dict]) -> str:
    header = ["run", "task", "mode", "model", "data", "sizes", "status",
              "meta U", "train obj", "test gap %", "evals", "min", "calls"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["run"], r["task"], r["mode"], r["model"], str(r["data"]),
            r["subtasks"] or "-", r["status"],
            _fmt(r["meta_utility"], ".3f"), _fmt(r["train_objective"], ".3f"),
            _fmt(r["test_gap"], ".3f"), _fmt(r["evals"]),
            _fmt(r["minutes"], ".1f"), _fmt(r["llm_calls"]),
        ]) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task")
    parser.add_argument("--sort", choices=["started", "meta", "objective", "test", "task"],
                        default="started")
    parser.add_argument("--unfinished", action="store_true",
                        help="show only runs with no summary.json")
    parser.add_argument("--csv", help="also write the table to this path")
    args = parser.parse_args(argv)

    rows = collect(args.task)
    if args.unfinished:
        rows = [r for r in rows if r["status"] != "ok"]
    if not rows:
        print(f"No runs under {os.path.relpath(RUNS, ROOT)}.")
        return 1

    keys = {
        "started": lambda r: r["started"],
        "task": lambda r: (r["task"], r["started"]),
        "meta": lambda r: (r["meta_utility"] is None, r["meta_utility"]),
        "objective": lambda r: (r["train_objective"] is None, r["train_objective"]),
        "test": lambda r: (r["test_gap"] is None, r["test_gap"]),
    }
    rows.sort(key=keys[args.sort])

    print(table(rows))
    finished = sum(r["status"] == "ok" for r in rows)
    print(f"\n{len(rows)} run(s), {finished} finished. "
          f"Logs: runs/llm/MoH/<task>/<timestamp>/run.log")
    if any(r["test_gap"] is None for r in rows):
        print("Runs with no test gap have not been benchmarked yet:\n"
              "  bash scripts/llm/MoH/benchmark.sh")

    if args.csv:
        out = args.csv if os.path.isabs(args.csv) else os.path.join(ROOT, args.csv)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written to {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
