"""An offline LLM client, for testing the plumbing without spending money.

It answers every prompt the way a well-behaved model would: a design idea in
braces followed by a fenced Python function with the name the prompt asked for.
The functions are deliberately trivial and randomly perturbed, so successive
calls produce distinct objectives and the MCTS tree actually branches.

Use it to check that a task, a config or a code change works end to end::

    cd solvers/llm/MCTS-AHD
    python main.py problem=atsp_gls llm_client=stub max_fe=12

Nothing it produces is a research result.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .base import BaseClient

_TEMPLATES = {
    "select_next_node": """\
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    best, best_score = None, float('inf')
    for node in unvisited_nodes:
        score = distance_matrix[current_node][node] + {w:.3f} * distance_matrix[node][destination_node]
        if score < best_score:
            best, best_score = node, score
    return best""",
    "heuristics": """\
def heuristics(distance_matrix):
    d = distance_matrix.astype(float)
    row_min = np.min(np.where(np.eye(d.shape[0], dtype=bool), np.inf, d), axis=1, keepdims=True)
    return d + {w:.3f} * (d - row_min)""",
    "arc_badness": """\
def arc_badness(distance_matrix, tour, penalty_count, iteration):
    u = np.asarray(tour)
    v = np.roll(u, -1)
    return distance_matrix[u, v] * (1.0 + {w:.3f} * np.arange(u.size) / max(1, u.size))""",
}


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


class StubClient(BaseClient):
    """Deterministic-by-seed fake model. Makes no network calls."""

    def __init__(self, model: str = "stub", temperature: float = 1.0, seed: int = 0):
        super().__init__(model, temperature)
        self._rng = random.Random(seed)

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        prompt = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))

        # Thought-alignment calls ask for prose, not code.
        if ("describe the Design Idea" in prompt
                or "re-describe the algorithm" in prompt):
            return [_Choice(_Message(
                "The heuristic scores each option by its direct cost and a small "
                "correction term, preferring cheap options while keeping a mild "
                "bias that varies with position."))] * n

        match = re.search(r"function named +\\?\s*'([A-Za-z_][A-Za-z_0-9]*)'", prompt)
        name = match.group(1) if match else "heuristics"
        base = name.split("_v")[0]
        template = _TEMPLATES.get(base, _TEMPLATES["heuristics"])
        body = template.format(w=self._rng.uniform(0.01, 0.9))
        text = ("{A stub heuristic that weights the direct cost against one "
                "correction term.}\n\n```python\nimport numpy as np\n"
                f"{body}\n```")
        return [_Choice(_Message(text)) for _ in range(n)]
