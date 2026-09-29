"""Step 5 — Condition B (RAG): Question -> Retriever -> top-k context -> LLM -> Answer.

    python -m src.experiments.run_rag [--config ...] [--limit N] [--overwrite]

Output: results/raw/rag_results.jsonl (+ rag_run_meta.json). Every record
stores the retrieved chunks (IDs, ranks, similarity scores, text) and the
exact prompt sent to the LLM.
"""
from src.experiments.runner import cli

if __name__ == "__main__":
    cli("rag", __doc__)
