import os
from pathlib import Path


from relevancedb import RelevanceDB

DOCS_DIR = Path(__file__).parent / "company_docs"
DATA_DIR = Path(__file__).parent / ".relevancedb_data"


def main():
    db = RelevanceDB(
        llm_model=os.getenv("RELEVANCEDB_LLM_MODEL", "gpt-4o-mini"),
        data_dir=DATA_DIR,
        verbose=True,
    )
    print(f"\n{db}\n")

    print("Ingesting documents...")
    summary = db.ingest(DOCS_DIR)
    print(f"\n{summary}\n")

    queries = [
        # WHO → graph head first — finds Alice via APPROVED relation
        "who approved the data retention policy?",

        # WHY → graph head — causal chain: audit → policy change
        "why was the data retention policy changed?",

        # WHAT + disambiguation — strawberry here = project, not fruit
        "what is the current status of the strawberry project?",

        # WHO → graph head — finds Bob as project owner
        "who owns the strawberry project?",

        # WHEN → timeline head — version + recency
        "when was the retention policy last updated?",

        # HOW → semantic + graph — implementation details
        "how were the new retention controls implemented?",
    ]

    for question in queries:
        print("-" * 60)
        print(f"Q: {question}")
        result = db.query(question)
        print(f"A: {result.answer}")
        print()
        print(result.explain())


if __name__ == "__main__":
    main()