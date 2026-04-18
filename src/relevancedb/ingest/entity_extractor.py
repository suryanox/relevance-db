"""
entity_extractor.py — zero-schema entity and relation extraction.

Uses an LLM (via litellm) to extract entities and relations from
a document chunk. The user provides no schema — we infer types
automatically and normalise relation names so duplicates collapse.

Output feeds directly into graph_store.py.

Normalisation: "MANAGES", "ORCHESTRATES", "HANDLES" all become
"MANAGES" via the LLM itself — we ask it to use canonical verbs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from relevancedb.ingest.loader import Document
from relevancedb.store.graph_store import Entity, Relation

CANONICAL_RELATIONS = [
    "OWNS", "MANAGES", "DEPENDS_ON", "CREATED", "UPDATED",
    "APPROVED", "REJECTED", "USES", "PART_OF", "RELATED_TO",
]

_SYSTEM_PROMPT = f"""
You are an entity and relation extractor. Given a text passage, extract:
1. Entities — named things: people, organisations, systems, policies, projects, concepts.
2. Relations — directional relationships between entities.

Rules:
- Only extract entities that are explicitly named in the text.
- Use these relation types where possible: {", ".join(CANONICAL_RELATIONS)}
- If none fit, invent a short SCREAMING_SNAKE_CASE verb.
- Do NOT extract generic words like "system", "data", "process" unless they are proper names.
- Return ONLY valid JSON. No explanation, no markdown, no code fences.

JSON format:
{{
  "entities": [
    {{"name": "Alice", "type": "PERSON"}},
    {{"name": "RetentionPolicy", "type": "POLICY"}}
  ],
  "relations": [
    {{"source": "Alice", "relation": "APPROVED", "target": "RetentionPolicy"}}
  ]
}}
""".strip()


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]

    def __repr__(self) -> str:
        return (
            f"ExtractionResult("
            f"entities={len(self.entities)}, "
            f"relations={len(self.relations)})"
        )


class EntityExtractor:
    """
    Extracts entities and relations from documents using an LLM.

    Args:
        llm_model: litellm-compatible model string.
                   e.g. "gpt-4o-mini", "claude-haiku-...", "ollama/mistral"
    """

    def __init__(self, llm_model: str = "gpt-4o-mini") -> None:
        self.llm_model = llm_model

    def extract(self, doc: Document) -> ExtractionResult:
        """
        Extract entities and relations from a full document.

        Splits into chunks internally to stay within context limits,
        then merges and deduplicates results.

        Args:
            doc: Parsed Document from loader.py.

        Returns:
            ExtractionResult with Entity and Relation lists.
        """
        # split into ~2000 char windows for extraction
        # (smaller than embedding chunks — LLM needs more context per window)
        windows = self._split_windows(doc.text, window_size=2000, overlap=200)

        all_entities: dict[str, Entity] = {}  # name → Entity, deduped
        all_relations: list[Relation] = []

        for window in windows:
            try:
                result = self._extract_window(window, doc_path=str(doc.path))
                for e in result.entities:
                    if e.name not in all_entities:
                        all_entities[e.name] = e
                all_relations.extend(result.relations)
            except Exception as exc:
                print(f"[relevancedb] entity extraction warning: {exc}")

        seen_relations: set[tuple] = set()
        deduped_relations = []
        for r in all_relations:
            key = (r.source, r.relation_type, r.target)
            if key not in seen_relations:
                seen_relations.add(key)
                deduped_relations.append(r)

        return ExtractionResult(
            entities=list(all_entities.values()),
            relations=deduped_relations,
        )


    def _extract_window(self, text: str, doc_path: str) -> ExtractionResult:
        """Send one text window to the LLM and parse the JSON response."""
        import litellm

        response = litellm.completion(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,        
            max_tokens=1024,
        )

        raw = response.choices[0].message.content.strip()
        parsed = self._parse_json(raw)

        entities = [
            Entity(
                name=e["name"].strip(),
                type=e.get("type", "UNKNOWN").upper(),
                doc_path=doc_path,
            )
            for e in parsed.get("entities", [])
            if e.get("name")
        ]

        relations = [
            Relation(
                source=r["source"].strip(),
                target=r["target"].strip(),
                relation_type=r.get("relation", "RELATED_TO").upper(),
                doc_path=doc_path,
            )
            for r in parsed.get("relations", [])
            if r.get("source") and r.get("target")
        ]

        return ExtractionResult(entities=entities, relations=relations)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        Parse LLM JSON output defensively.
        Strips markdown fences if the model adds them despite instructions.
        """
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw: {raw[:200]}")

    @staticmethod
    def _split_windows(text: str, window_size: int, overlap: int) -> list[str]:
        """Split text into overlapping windows for extraction."""
        if len(text) <= window_size:
            return [text]

        windows = []
        start = 0
        while start < len(text):
            end = start + window_size
            windows.append(text[start:end])
            start += window_size - overlap

        return windows