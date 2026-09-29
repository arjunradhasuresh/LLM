#!/usr/bin/env bash
# Run the complete experiment end-to-end (from the repository root).
# Each step can also be run on its own; see README.md.
set -euo pipefail
CONFIG="${1:-config/config.yaml}"

python -m src.data.prepare           --config "$CONFIG"
python -m src.data.build_corpus      --config "$CONFIG"
python -m src.retrieval.build_index  --config "$CONFIG"
python -m src.experiments.run_no_rag --config "$CONFIG"
python -m src.experiments.run_rag    --config "$CONFIG"
python -m src.evaluation.evaluate    --config "$CONFIG"
python -m src.analysis.generate_results --config "$CONFIG"
