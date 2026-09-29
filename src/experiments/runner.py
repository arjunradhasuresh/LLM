"""Shared experiment loop for the No-RAG and RAG conditions.

The two conditions differ in exactly one thing: whether retrieved context is
inserted into the prompt. Questions, system prompt, model, temperature,
max_tokens and seed are identical (taken from the same config).

Raw outputs are appended to results/raw/<condition>_results.jsonl one
record per question, immediately after each API call, so a crash or rate
limit never loses finished work. Re-running the script resumes: questions
already answered successfully with the *same* settings are skipped; failed
ones are retried. If the settings changed, the script refuses to mix runs
unless `--overwrite` is given.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from src.config import fingerprint, generation_settings, public_config, resolve_path
from src.llm.client import LLMError, FatalLLMError, make_client
from src.prompts import build_no_rag_prompt, build_rag_prompt, load_template
from src.retrieval.build_index import index_settings
from src.utils import append_jsonl, get_logger, read_jsonl, utc_now, write_json

log = get_logger(__name__)
CONDITIONS = ("no_rag", "rag")


def results_path(cfg: dict[str, Any], condition: str) -> Path:
    return resolve_path(cfg["paths"]["results_raw"]) / f"{condition}_results.jsonl"


def load_questions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve_path(cfg["dataset"]["processed_path"])
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — run `python -m src.data.prepare` first.")
    questions = read_jsonl(path)
    if not questions:
        raise ValueError(f"{path} contains no questions.")
    if len(questions) != cfg["dataset"]["number_of_questions"]:
        raise ValueError(
            f"{path} has {len(questions)} questions but config says "
            f"{cfg['dataset']['number_of_questions']} — re-run `python -m src.data.prepare`."
        )
    return questions


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    """Latest record per question_id (later lines supersede earlier retries)."""
    if not path.is_file():
        return {}
    return {r["question_id"]: r for r in read_jsonl(path)}


def run_fingerprint(cfg: dict[str, Any], condition: str, questions: list[dict[str, Any]],
                    system_prompt: str, template: str) -> str:
    settings = {
        "condition": condition,
        "generation": generation_settings(cfg),
        "system_prompt": system_prompt,
        "template": template,
        "question_ids": [q["question_id"] for q in questions],
    }
    if condition == "rag":
        settings["retrieval"] = {**index_settings(cfg), "top_k": cfg["retrieval"]["top_k"]}
    return fingerprint(settings)


def run_condition(cfg: dict[str, Any], condition: str, client=None, retriever=None,
                  overwrite: bool = False, limit: int | None = None) -> dict[str, int]:
    """Run one condition over the fixed question set. Returns counts."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    questions = load_questions(cfg)
    if limit:
        questions = questions[:limit]
    system_prompt = load_template(cfg, "system")
    template = load_template(cfg, condition)
    run_id = run_fingerprint(cfg, condition, load_questions(cfg), system_prompt, template)
    gen = generation_settings(cfg)

    out_path = results_path(cfg, condition)
    if overwrite and out_path.exists():
        backup = out_path.with_suffix(f".jsonl.bak_{utc_now().replace(':', '')}")
        out_path.rename(backup)
        log.info("Moved previous results to %s", backup)
    existing = latest_records(out_path)
    other_runs = {r["run_id"] for r in existing.values()} - {run_id}
    if other_runs:
        raise RuntimeError(
            f"{out_path} contains results from a run with different settings ({other_runs}). "
            "Use --overwrite to start a fresh run (old file is kept as a backup)."
        )
    done = {qid for qid, r in existing.items() if r.get("status") == "ok"}
    todo = [q for q in questions if q["question_id"] not in done]
    log.info("[%s] run_id=%s: %d questions, %d already done, %d to run",
             condition, run_id, len(questions), len(done & {q['question_id'] for q in questions}), len(todo))
    if not todo:
        return {"ok": len(done), "error": 0, "skipped": len(done)}

    client = client or make_client(cfg["llm"], seed=gen["seed"])
    if condition == "rag" and retriever is None:
        from src.retrieval.retriever import Retriever
        retriever = Retriever.from_config(cfg)

    write_json(out_path.with_name(f"{condition}_run_meta.json"), {
        "run_id": run_id, "condition": condition, "started_at": utc_now(),
        "generation_settings": gen, "system_prompt": system_prompt, "prompt_template": template,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "config": public_config(cfg),
    })

    counts = {"ok": 0, "error": 0, "skipped": len(done)}
    for i, q in enumerate(todo, start=1):
        record: dict[str, Any] = {
            "run_id": run_id, "condition": condition, "question_id": q["question_id"],
            "question": q["question"], "model_requested": gen["model"], **{
                k: gen[k] for k in ("provider", "temperature", "max_tokens", "seed")},
            "system_prompt": system_prompt, "timestamp": utc_now(),
        }
        try:
            if condition == "rag":
                retrieved = retriever.retrieve(q["question"])
                record["retrieved"] = retrieved
                user_prompt = build_rag_prompt(template, q["question"], retrieved)
            else:
                user_prompt = build_no_rag_prompt(template, q["question"])
            record["user_prompt"] = user_prompt
            resp = client.complete(system_prompt, user_prompt)
            record.update(
                status="ok", answer=resp.text, model_returned=resp.model,
                finish_reason=resp.finish_reason, usage=resp.usage,
                system_fingerprint=resp.system_fingerprint, latency_seconds=resp.latency_seconds,
            )
            counts["ok"] += 1
        except FatalLLMError:
            raise  # no point continuing without credentials
        except Exception as e:  # recorded, reported, retried on next run
            record.update(status="error", answer=None, error=f"{type(e).__name__}: {e}")
            counts["error"] += 1
            log.error("[%s] %s failed: %s", condition, q["question_id"], record["error"])
        append_jsonl(out_path, record)
        if i % 10 == 0 or i == len(todo):
            log.info("[%s] %d/%d done", condition, i, len(todo))

    log.info("[%s] finished: %d ok, %d errors, %d previously done -> %s",
             condition, counts["ok"], counts["error"], counts["skipped"], out_path)
    if counts["error"]:
        log.warning("Some questions failed. Re-run the same command to retry only those.")
    return counts


def cli(condition: str, description: str) -> None:
    from src.config import base_arg_parser, load_config
    parser = base_arg_parser(description)
    parser.add_argument("--overwrite", action="store_true",
                        help="Start a fresh run (existing results are backed up)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (smoke test)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        counts = run_condition(cfg, condition, overwrite=args.overwrite, limit=args.limit)
    except (LLMError, FileNotFoundError, RuntimeError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)
    sys.exit(1 if counts["error"] else 0)
