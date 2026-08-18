"""An offline LLM, for testing the plumbing without spending money.

MoH asks its model for four different things, and a stub is only useful if it
answers all four the way a well-behaved model would:

1. **directions / insights** — a ``json`` code block with a ``direction`` or
   ``insights`` list, which both loops parse with ``json.loads``;
2. **a heuristic** — an idea in braces followed by a fenced Python function
   with the name the task's prompt asked for;
3. **a heuristic-optimizer** — a fenced ``improve_algorithm`` that calls the
   language model and the utility function, because ``moh.py`` rejects any
   candidate optimizer that never prompts (that check is what stops the outer
   loop drifting into hand-written local edits);
4. **an iteration count** — the loop-depth audit in
   ``prompts/helper/check_iter.txt``, answered in a ``txt`` block.

Everything it returns is deliberately trivial and seeded-random, so successive
calls differ, populations actually fill, and the two loops both branch.

    cd solvers/llm/MoH
    python main.py problem=atsp_gls llm_client@heu=stub llm_client@meta=stub \
        n_iterations=1 pop_size=3 max_eval_calls=12

Nothing it produces is a research result.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .base import BaseClient

_HEURISTICS = {
    "update_edge_distance": """\
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    return updated_edge_distance * (1.0 + {w:.3f}) / (1.0 + edge_n_used)""",
    "select_next_node": """\
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    best, best_score = None, float('inf')
    for node in unvisited_nodes:
        score = distance_matrix[current_node][node] + {w:.3f} * distance_matrix[node][destination_node]
        if score < best_score:
            best, best_score = node, score
    return best""",
    "arc_badness": """\
def arc_badness(distance_matrix, tour, penalty_count, iteration):
    u = np.asarray(tour)
    v = np.roll(u, -1)
    return distance_matrix[u, v].astype(float) * (1.0 + {w:.3f})""",
}

_OPTIMIZER = '''\
def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = "You are an expert in optimization heuristics."
    selected = population.get_random_solution(task)
    directions = language_model.prompt(expertise, "Suggest one direction.", 0.7)
    responses = language_model.prompt_batch(
        expertise,
        ["Improve this under {{}}: ".format(function_format) + str(selected['best_sol'])
         for _ in range({k})],
        temperature={t:.2f})
    scored = []
    for response in responses:
        code = extract_code(response)
        idea = extract_idea(response)
        if not code:
            continue
        scored.append((idea, code, utility(code, idea, task)))
    if not scored:
        best = population.get_solution_by_index(task, 0)
        return best['idea'], best['best_sol'], best['utility']
    return min(scored, key=lambda item: item[2])'''


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


class StubClient(BaseClient):
    """Deterministic-by-seed fake model. Makes no network calls."""

    def __init__(self, model: str = "stub", temperature: float = 1.0,
                 batch_size: int = 5, seed: int = 0, role: str = "heu",
                 cache_dir: str | None = None, **_ignored):
        super().__init__(model, temperature, batch_size, cache_dir=cache_dir,
                         max_retries=1, role=role)
        self._rng = random.Random(seed)

    # The stub answers instantly and never fails, so it bypasses the retry
    # path entirely and formats a response straight from the prompt text.
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        prompt = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
        return [_Choice(_Message(self._answer(prompt))) for _ in range(n)]

    def _answer(self, prompt: str) -> str:
        # Dispatch on what the prompt is ASKING FOR, which lives in its last few
        # hundred characters, not on what it happens to quote. MoH's prompts
        # embed whole functions as context — a request to improve an optimizer
        # quotes source that itself contains `["insights"]` and ```json — so
        # matching anywhere in the prompt picks the wrong branch and the run
        # silently degrades into rejected candidates.
        tail = prompt[-800:]

        # 4. loop-depth audit
        if "number of iterations the outer loop" in prompt:
            return "The outer loop runs once.\n```txt\n1\n```"

        # 1. structured direction / insight lists — always asked for as JSON
        if "json" in tail.lower():
            if '"direction"' in tail or "'direction'" in tail:
                return self._json_block("direction", "Refinement: scale the seed rule.")
            return self._json_block("insights", "Try a different weighting of the cost term.")

        # 3. a heuristic-optimizer (the outer loop). Only the meta prompt names
        # `improve_algorithm`, so this cannot fire for an inner-loop request.
        if "improve_algorithm" in prompt:
            body = _OPTIMIZER.format(k=self._rng.randint(1, 2),
                                     t=self._rng.uniform(0.5, 1.0))
            return ("{A stub optimizer that samples one parent and asks for a "
                    "couple of variants.}\n\n```python\n"
                    "from utils.utils import extract_code, extract_idea\n"
                    "import json\n\n" + body + "\n```")

        # 2. a heuristic (the inner loop)
        name = self._entry_point(prompt)
        body = _HEURISTICS[name].format(w=self._rng.uniform(0.01, 0.9))
        return ("{A stub heuristic that rescales the seed rule by a constant.}\n\n"
                "```python\nimport numpy as np\n" + body + "\n```")

    def _json_block(self, key: str, first: str) -> str:
        items = [first, "Innovation: combine row and column minima.",
                 f"Refinement: weight the term by {self._rng.uniform(0.1, 2.0):.2f}."]
        body = ",\n    ".join(
            (f'{{"content": "{item}"}}' if key == "direction" else f'"{item}"')
            for item in items)
        return f'```json\n{{\n  "{key}": [\n    {body}\n  ]\n}}\n```'

    @staticmethod
    def _entry_point(prompt: str) -> str:
        for name in _HEURISTICS:
            if name in prompt:
                return name
        match = re.search(r"function named +\\?\s*'?([A-Za-z_][A-Za-z_0-9]*)'?", prompt)
        if match and match.group(1) in _HEURISTICS:
            return match.group(1)
        return "update_edge_distance"
