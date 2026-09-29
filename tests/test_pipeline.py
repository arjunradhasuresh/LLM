"""End-to-end pipeline with the offline mock LLM + fake embedder (no network)."""
import json

import pytest

from src.analysis.generate_results import generate
from src.config import ConfigError, load_config, validate_config
from src.evaluation.evaluate import evaluate
from src.experiments.runner import results_path, run_condition
from src.llm.client import LLMError, LLMResponse, MissingAPIKeyError, OpenAIChatClient
from src.retrieval.build_index import build_index
from src.retrieval.retriever import Retriever
from src.utils import read_jsonl


class FlakyClient:
    """Fails on one specific question, answers the rest."""

    def complete(self, system, user, json_mode=False):
        if "veins" in user.lower():
            raise LLMError("simulated API failure")
        return LLMResponse(text="Canberra.", model="fake-model", finish_reason="stop")


def test_full_pipeline_with_mock(tmp_cfg, fake_embedder):
    build_index(tmp_cfg, embedder=fake_embedder)
    retriever = Retriever.from_config(tmp_cfg, embedder=fake_embedder)
    assert run_condition(tmp_cfg, "no_rag")["ok"] == 3
    assert run_condition(tmp_cfg, "rag", retriever=retriever)["ok"] == 3

    rag = read_jsonl(results_path(tmp_cfg, "rag"))
    no_rag = read_jsonl(results_path(tmp_cfg, "no_rag"))
    assert [r["question_id"] for r in rag] == [r["question_id"] for r in no_rag]
    assert all(len(r["retrieved"]) == 2 and "Context:" in r["user_prompt"] for r in rag)
    assert all("Context:" not in r["user_prompt"] for r in no_rag)
    assert rag[0]["system_prompt"] == no_rag[0]["system_prompt"]
    assert rag[0]["temperature"] == no_rag[0]["temperature"]

    # resume: nothing left to do
    assert run_condition(tmp_cfg, "no_rag")["skipped"] == 3

    meta = evaluate(tmp_cfg, semantic_embedder=fake_embedder)
    assert meta["n_paired"] == 3 and meta["n_judge_errors"] == 0
    summary = generate(tmp_cfg)
    assert summary["MOCK_RESULTS_NOT_REAL"] is True
    assert summary["primary_strict"]["n"] == 3
    tables = tmp_cfg["paths"]["results_tables"]
    for name in ("combined_results.csv", "summary.csv", "evaluation_results.csv",
                 "retrieval_results.csv", "retrieval_vs_correctness.csv", "no_rag_results.csv"):
        assert (__import__("pathlib").Path(tables) / name).is_file()
    assert (__import__("pathlib").Path(tmp_cfg["paths"]["results_figures"]) / "fig1_accuracy_comparison.png").is_file()

    # judge outputs are cached: a second evaluation makes no new calls
    assert evaluate(tmp_cfg, semantic_embedder=fake_embedder)["new_judge_calls"] == 0


def test_failed_request_is_recorded_not_hidden(tmp_cfg):
    counts = run_condition(tmp_cfg, "no_rag", client=FlakyClient())
    assert counts == {"ok": 2, "error": 1, "skipped": 0}
    recs = {r["question_id"]: r for r in read_jsonl(results_path(tmp_cfg, "no_rag"))}
    assert recs["tqa_002"]["status"] == "error" and "simulated" in recs["tqa_002"]["error"]
    # re-run retries only the failed question
    counts2 = run_condition(tmp_cfg, "no_rag")
    assert counts2["ok"] == 1 and counts2["skipped"] == 2


def test_settings_change_refuses_to_mix_runs(tmp_cfg):
    run_condition(tmp_cfg, "no_rag")
    tmp_cfg["llm"]["temperature"] = 0.7
    with pytest.raises(RuntimeError, match="different settings"):
        run_condition(tmp_cfg, "no_rag")
    run_condition(tmp_cfg, "no_rag", overwrite=True)  # explicit fresh run works


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY_TEST_UNSET", raising=False)
    with pytest.raises(MissingAPIKeyError):
        OpenAIChatClient(model="x", temperature=0, max_tokens=5, api_key_env="OPENAI_API_KEY_TEST_UNSET")


def test_invalid_config_rejected():
    cfg = load_config()
    cfg["retrieval"]["chunk_overlap"] = cfg["retrieval"]["chunk_size"]
    with pytest.raises(ConfigError):
        validate_config(cfg)
    cfg = load_config()
    cfg["llm"]["provider"] = "nonexistent"
    with pytest.raises(ConfigError):
        validate_config(cfg)
    cfg = load_config()
    del cfg["dataset"]["number_of_questions"]
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_openai_client_mocked(monkeypatch):
    """OpenAI client parses a response without touching the network."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIChatClient(model="gpt-test", temperature=0, max_tokens=5, seed=1)

    class _Msg:
        content = " Canberra. "

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        model = "gpt-test-2025"
        usage = None
        system_fingerprint = "fp_1"

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
    r = client.complete("sys", "user", json_mode=True)
    assert r.text == "Canberra." and r.model == "gpt-test-2025"
    assert captured["seed"] == 1 and captured["temperature"] == 0
    assert captured["response_format"] == {"type": "json_object"}
    json.dumps(captured["messages"])  # serialisable


def test_no_credits_error_stops_immediately(monkeypatch, tmp_cfg):
    """HTTP 429 'insufficient_quota' must abort at once instead of retrying."""
    import httpx
    import openai
    from src.llm.client import QuotaExceededError
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIChatClient(model="gpt-test", temperature=0, max_tokens=5, max_retries=5)
    calls = []

    def fake_create(**kwargs):
        calls.append(1)
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        raise openai.RateLimitError("Error code: 429", response=httpx.Response(429, request=req),
                                    body={"message": "no credits", "type": "insufficient_quota",
                                          "code": "credit_balance_exhausted"})

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
    with pytest.raises(QuotaExceededError, match="no credits"):
        client.complete("sys", "user")
    assert len(calls) == 1
    with pytest.raises(QuotaExceededError):
        run_condition(tmp_cfg, "no_rag", client=client)
