"""
intent_classifier.py — query intent detection before retrieval.

This is the second novel piece in RelevanceDB.

The problem with flat vector search:
  Every query is treated identically — embed the question, find
  similar vectors. But "what is the retention policy?" and
  "why was the retention policy changed?" need completely different
  retrieval strategies.

  WHAT → semantic head (find the definition/description)
  WHY  → graph head (traverse causal chains, decisions, approvals)
  WHEN → timeline head (find the version that existed at a point in time)
  WHO  → graph head (find PERSON entities and their relations)
  HOW  → semantic head + graph head (find process + relationships)

We classify intent first, then the query_planner uses it to decide
which heads to query and in what order.

Classification is done with a lightweight LLM call — cheap, fast,
single token output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    WHAT = "what"    # definition, description, fact
    WHY  = "why"     # reason, cause, decision, rationale
    WHEN = "when"    # time, date, version, history
    WHO  = "who"     # person, team, owner, approver
    HOW  = "how"     # process, steps, mechanism
    UNKNOWN = "unknown"  # fallback — hit all heads


INTENT_HEAD_MAP: dict[Intent, list[str]] = {
    Intent.WHAT:    ["semantic", "graph"],
    Intent.WHY:     ["graph", "semantic"],
    Intent.WHEN:    ["timeline", "semantic"],
    Intent.WHO:     ["graph", "semantic"],
    Intent.HOW:     ["semantic", "graph"],
    Intent.UNKNOWN: ["semantic", "graph", "timeline"],
}

_SYSTEM_PROMPT = """
You classify the intent of a retrieval query into exactly one of these categories:

  what   — asks for a definition, description, or fact
  why    — asks for a reason, cause, decision, or rationale
  when   — asks about time, dates, history, or versions
  who    — asks about a person, team, owner, or approver
  how    — asks about a process, steps, or mechanism
  unknown — none of the above

Reply with a single word only. No punctuation, no explanation.
""".strip()


@dataclass
class ClassifiedQuery:
    question: str
    intent: Intent
    heads: list[str]     

    def __repr__(self) -> str:
        return (
            f"ClassifiedQuery("
            f"intent={self.intent.value!r}, "
            f"heads={self.heads}, "
            f"question={self.question[:50]!r})"
        )


class IntentClassifier:
    """
    Classifies a natural-language query into an Intent.

    Uses a fast LLM call with a single-token response.
    Falls back to rule-based heuristics if the LLM fails.

    Args:
        llm_model: litellm model string.
    """

    def __init__(self, llm_model: str = "gpt-4o-mini") -> None:
        self.llm_model = llm_model

    def classify(self, question: str) -> ClassifiedQuery:
        """
        Classify the intent of a query.

        Tries LLM first, falls back to keyword heuristics.

        Args:
            question: Natural-language query string.

        Returns:
            ClassifiedQuery with intent and ordered head list.
        """
        intent = self._classify_llm(question)
        heads = INTENT_HEAD_MAP[intent]
        return ClassifiedQuery(question=question, intent=intent, heads=heads)

    
    def _classify_llm(self, question: str) -> Intent:
        """Try LLM classification, fall back to heuristics on failure."""
        try:
            import litellm
            response = litellm.completion(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=5,   # single word response
            )
            raw = response.choices[0].message.content.strip().lower()
            return self._parse_intent(raw)
        except Exception:
            # LLM unavailable or no API key — fall back to keywords
            return self._classify_heuristic(question)

    @staticmethod
    def _parse_intent(raw: str) -> Intent:
        """Map LLM output string to Intent enum."""
        mapping = {i.value: i for i in Intent}
        # take first word in case model adds punctuation
        word = raw.split()[0].strip(".,?!") if raw else "unknown"
        return mapping.get(word, Intent.UNKNOWN)

    @staticmethod
    def _classify_heuristic(question: str) -> Intent:
        """
        Keyword-based fallback. Good enough for ~80% of real queries.
        Used when LLM is unavailable.
        """
        q = question.lower().strip()

        if any(q.startswith(w) for w in ("why", "what caused", "reason", "rationale")):
            return Intent.WHY
        if any(q.startswith(w) for w in ("who", "which person", "which team")):
            return Intent.WHO
        if any(q.startswith(w) for w in ("when", "what date", "what time", "since when")):
            return Intent.WHEN
        if any(q.startswith(w) for w in ("how", "what steps", "what process")):
            return Intent.HOW
        if any(q.startswith(w) for w in ("what", "which", "describe", "define", "tell me about")):
            return Intent.WHAT

        return Intent.UNKNOWN