"""OpenAI-compatible client. The ATSP benchmark standardises on gpt-4o-mini.

The MoH paper uses ``gpt-4o-mini (2024-07-18)`` as its base LLM for every
headline result (Section 4, "Training and Inference"), which is also what the
EoH, ReEvo, HSEvo and MCTS-AHD sides of this repository use — so the model is
held constant and the comparison is between the search methods.

``base_url`` makes this work against any OpenAI-compatible endpoint (vLLM,
Ollama, LM Studio); see ``configs/llm/MoH/cfg/llm_client/local.yaml``.
"""

import logging
from typing import Optional

from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

logger = logging.getLogger(__name__)


class OpenAIClient(BaseClient):
    """Hydra ``_target_``: ``utils.llm_client.openai.OpenAIClient``."""

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        batch_size: int = 5,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 6,
        role: str = "heu",
    ) -> None:
        super().__init__(model, temperature, batch_size, cache_dir=cache_dir,
                         max_retries=max_retries, role=role)

        if OpenAI is None:
            logger.fatal("Package `openai` is required: "
                         "pip install -r envs/llm/MoH/requirements.txt")
            raise ImportError("openai")
        if not api_key and not base_url:
            raise ValueError(
                "no OPENAI_API_KEY. Put it in envs/.env (cp envs/.env.example "
                "envs/.env), or run with llm_client@heu=stub llm_client@meta=stub "
                "for a free offline dry run.")

        # The SDK's own retry loop is disabled: BaseClient already backs off,
        # and two nested retry loops turn one outage into minutes of silence.
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             timeout=timeout, max_retries=0)

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            stream=False,
        )
        return response.choices
