from __future__ import annotations
import os

# Hardcoded — fast, 33M params, runs fully locally, no API key.
# User never needs to know or change this.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class ModelConfig:
    """
    Holds model configuration for RelevanceDB.

    Only the LLM model is user-facing. Embedding is an internal detail.

    Precedence for llm_model:
      1. Explicit argument to RelevanceDB()
      2. RELEVANCEDB_LLM_MODEL environment variable
      3. Raises — no silent default that could surprise with API charges
    """

    def __init__(self, llm_model: str | None = None) -> None:
        self.llm_model = self._resolve_llm(llm_model)
        self.embed_model = EMBED_MODEL

    @staticmethod
    def _resolve_llm(explicit: str | None) -> str:
        model = explicit or os.getenv("RELEVANCEDB_LLM_MODEL")

        if not model:
            raise ValueError(
                "No LLM model specified.\n\n"
                "Pass it explicitly:\n"
                "    db = RelevanceDB(llm_model='gpt-4o-mini')\n\n"
                "Or set the environment variable:\n"
                "    export RELEVANCEDB_LLM_MODEL=gpt-4o-mini\n\n"
                "Accepts any litellm-compatible model string.\n"
                "See https://docs.litellm.ai/docs/providers"
            )

        return model

    def __repr__(self) -> str:
        return f"ModelConfig(llm={self.llm_model!r})"