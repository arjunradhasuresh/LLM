"""Step 4 — Condition A (No-RAG): Question -> LLM -> Answer.

    python -m src.experiments.run_no_rag [--config ...] [--limit N] [--overwrite]

Output: results/raw/no_rag_results.jsonl (+ no_rag_run_meta.json)
"""
from src.experiments.runner import cli

if __name__ == "__main__":
    cli("no_rag", __doc__)
