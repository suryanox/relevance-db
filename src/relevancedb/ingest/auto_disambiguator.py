"""
auto_disambiguator.py — sense detection at index time.

This is the core novel contribution of RelevanceDB.

The problem:
  VectorDBs embed "strawberry" identically whether it refers to
  Apple's internal project or the fruit. At query time there is no
  way to separate them — the damage is done at ingest.

The fix:
  Before embedding, we read the surrounding context and ask an LLM:
  "what does this term mean HERE, in this specific passage?"
  The answer becomes a sense_id. Chunks are then stored in a
  namespace keyed by that sense_id.

  "strawberry" → sense_id="strawberry__apple_project"
  "strawberry" → sense_id="strawberry__fruit"

  Cosine search is then physically scoped to one namespace.
  The wrong strawberry cannot appear in results — ever.

Flow:
  chunk text → find ambiguous terms → resolve sense per term
            → assign namespace → semantic_store.add(namespace=sense_id)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from relevancedb.ingest.chunker import Chunk

_SYSTEM_PROMPT = """
You are a word sense disambiguation expert.

Given a text passage and a list of potentially ambiguous terms found in it,
determine the specific sense of each term as used IN THIS PASSAGE ONLY.

Return a JSON object mapping each term to a short snake_case sense label.
The label must describe what the term refers to in this specific context.

Rules:
- Be specific: not "project" but "software_project" or "codename"
- Be consistent: the same real-world thing should always get the same label
- If a term is unambiguous in context, still return a label (e.g. "person_name")
- Return ONLY valid JSON. No explanation, no markdown fences.

Example input terms: ["apple", "python"]
Example passage: "The Apple keynote introduced new M3 chips. Our Python scripts automate the build."
Example output:
{
  "apple": "tech_company",
  "python": "programming_language"
}
""".strip()


@dataclass
class DisambiguationResult:
    """Maps term → sense_id for a single chunk."""
    chunk_index: int
    senses: dict[str, str]   # term → sense_id e.g. {"strawberry": "apple_project"}
    namespace: str            # the primary namespace assigned to this chunk


class AutoDisambiguator:
    """
    Detects entity senses at index time and assigns namespaces to chunks.

    Args:
        llm_model:       litellm model string.
        min_term_length: Ignore terms shorter than this. Avoids disambiguating
                         common short words like "it", "as", "on".
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        min_term_length: int = 4,
    ) -> None:
        self.llm_model = llm_model
        self.min_term_length = min_term_length

        self._cache: dict[tuple[str, str], str] = {}

    def assign_namespaces(
        self,
        chunks: list[Chunk],
        candidate_terms: list[str],
    ) -> list[DisambiguationResult]:
        """
        For each chunk, resolve the sense of each candidate term and
        assign a namespace.

        Args:
            chunks:          Chunks from chunker.py.
            candidate_terms: Terms to disambiguate. Typically entity names
                             returned by entity_extractor.py.

        Returns:
            One DisambiguationResult per chunk, with senses + namespace.
        """
        results = []

        for chunk in chunks:
            # only disambiguate terms that actually appear in this chunk
            present = [
                t for t in candidate_terms
                if t.lower() in chunk.text.lower()
                and len(t) >= self.min_term_length
            ]

            if not present:
                results.append(DisambiguationResult(
                    chunk_index=chunk.chunk_index,
                    senses={},
                    namespace="default",
                ))
                continue

            senses = self._resolve_senses(chunk.text, present)
            namespace = self._pick_namespace(senses)

            results.append(DisambiguationResult(
                chunk_index=chunk.chunk_index,
                senses=senses,
                namespace=namespace,
            ))

        return results

  
    def _resolve_senses(self, text: str, terms: list[str]) -> dict[str, str]:
        """
        Ask the LLM to resolve the sense of each term in this specific text.
        Uses cache to avoid duplicate LLM calls.
        """
        fingerprint = text[:120].strip()
        to_resolve = []

        senses: dict[str, str] = {}
        for term in terms:
            cache_key = (term.lower(), fingerprint)
            if cache_key in self._cache:
                senses[term] = self._cache[cache_key]
            else:
                to_resolve.append(term)

        if not to_resolve:
            return senses

        try:
            resolved = self._call_llm(text, to_resolve)
            for term, sense in resolved.items():
                cache_key = (term.lower(), fingerprint)
                self._cache[cache_key] = sense
                senses[term] = sense
        except Exception as exc:
            print(f"[relevancedb] disambiguation warning: {exc}")
            for term in to_resolve:
                senses[term] = term.lower().replace(" ", "_")

        return senses

    def _call_llm(self, text: str, terms: list[str]) -> dict[str, str]:
        """Send text + terms to LLM, return term → sense_id mapping."""
        import litellm

        user_msg = (
            f"Terms to disambiguate: {json.dumps(terms)}\n\n"
            f"Passage:\n{text[:1500]}"
        )

        response = litellm.completion(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_json(raw)

    @staticmethod
    def _pick_namespace(senses: dict[str, str]) -> str:
        """
        Derive a single namespace string from the resolved senses.

        If multiple terms were disambiguated, use the first one.
        Namespace format: "{term}__{sense_id}"
        e.g. "strawberry__apple_project"
        """
        if not senses:
            return "default"
        term, sense = next(iter(senses.items()))
        safe_term = term.lower().replace(" ", "_")
        safe_sense = sense.lower().replace(" ", "_")
        return f"{safe_term}__{safe_sense}"

    @staticmethod
    def _parse_json(raw: str) -> dict[str, str]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        try:
            result = json.loads(cleaned.strip())
            return {k: str(v) for k, v in result.items()}
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw: {raw[:200]}")