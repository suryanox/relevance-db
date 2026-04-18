"""
config.py — model configuration for RelevanceDB.

Makes LLM and embedding model selection explicit.
No silent defaults that could surprise users with unexpected API calls.

Supported LLM providers (via litellm):
  OpenAI:    "gpt-4o-mini", "gpt-4o"
  Anthropic: "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"
  Ollama:    "ollama/mistral", "ollama/llama3", "ollama/phi3"
  Groq:      "groq/llama3-8b-8192"
  Any other litellm-compatible string works too.

Supported embed models (sentence-transformers, run locally):
  "BAAI/bge-small-en-v1.5"   — fast, 33M params, good for most use cases
  "BAAI/bge-base-en-v1.5"    — better quality, 109M params
  "BAAI/bge-large-en-v1.5"   — best quality, 335M params
  "all-MiniLM-L6-v2"         — lightest, good for dev/testing
"""

from __future__ import annotations

import os

OLLAMA_MODELS = {
    "ollama/mistral",
    "ollama/llama3",
    "ollama/phi3",
    "ollama/gemma2",
}


class ModelConfig:
    """
    Holds LLM and embedding model configuration.

    Precedence for llm_model:
      1. Explicit argument to RelevanceDB()
      2. RELEVANCEDB_LLM_MODEL environment variable
      3. Raises — no silent default

    Precedence for embed_model:
      1. Explicit argument to RelevanceDB()
      2. RELEVANCEDB_EMBED_MODEL environment variable
      3. "BAAI/bge-small-en-v1.5" — safe local default (no API key needed)
    """

    def __init__(
        self,
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.llm_model = self._resolve_llm(llm_model)
        self.embed_model = self._resolve_embed(embed_model)

    # ------------------------------------------------------------------

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
                "For fully local (no API key):\n"
                "    db = RelevanceDB(llm_model='ollama/mistral')\n\n"
                "Supported: any litellm-compatible model string.\n"
                "See https://docs.litellm.ai/docs/providers"
            )

        return model

    @staticmethod
    def _resolve_embed(explicit: str | None) -> str:
        # embed model has a safe local default — no API key needed
        return (
            explicit
            or os.getenv("RELEVANCEDB_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        )

    def is_local(self) -> bool:
        """True if the LLM runs fully locally via Ollama."""
        return self.llm_model in OLLAMA_MODELS

    def __repr__(self) -> str:
        local = " (local)" if self.is_local() else ""
        return (
            f"ModelConfig("
            f"llm={self.llm_model!r}{local}, "
            f"embed={self.embed_model!r})"
        )