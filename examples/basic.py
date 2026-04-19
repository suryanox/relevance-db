import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from relevancedb import RelevanceDB

DOCS = Path(__file__).parent / "company_docs"
db = RelevanceDB(llm_model="gpt-4o-mini")

db.ingest(DOCS)

questions = [
    "who approved the data retention policy?",
    "why was the retention policy changed?",
    "what is the current status of the strawberry project?",
    "who owns the strawberry project?",
    "when was the policy last updated?",
    "who is responsible for implementing the retention controls?",
]

for q in questions:
    print(f"Q: {q}")
    result = db.query(q)
    print(f"A: {result.answer}")
    print(result.explain())
    print("-" * 50)