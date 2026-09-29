"""Minimal chat-completion client used by both conditions and by the judge.

Providers
  openai  — OpenAI Chat Completions API (or any OpenAI-compatible server via
            `base_url`, e.g. Ollama / vLLM). API key read from the environment
            variable named in the config (default OPENAI_API_KEY).
  mock    — deterministic offline stand-in used for tests and dry runs. Its
            outputs are NOT real model answers; results produced with it are
            watermarked as MOCK by the analysis scripts.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from src.utils import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    """A request failed permanently (after retries) or returned no content."""


class FatalLLMError(LLMError):
    """An error where retrying cannot help (stop the whole run)."""


class MissingAPIKeyError(FatalLLMError):
    pass


class QuotaExceededError(FatalLLMError):
    """The API account has no credits / exceeded its billing quota."""


QUOTA_CODES = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}


def _is_quota_error(err: Exception) -> bool:
    body = getattr(err, "body", None) or {}
    if isinstance(body, dict) and "error" in body and isinstance(body["error"], dict):
        body = body["error"]
    codes = {str(body.get("code")), str(body.get("type"))} if isinstance(body, dict) else set()
    return bool(codes & QUOTA_CODES) or "insufficient_quota" in str(err)


@dataclass
class LLMResponse:
    text: str
    model: str                     # model name reported by the API
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    system_fingerprint: str | None = None
    latency_seconds: float = 0.0


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # python-dotenv is optional
        pass


class OpenAIChatClient:
    def __init__(self, model: str, temperature: float, max_tokens: int, seed: int | None = None,
                 api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None,
                 timeout: float = 60, max_retries: int = 5):
        _load_dotenv()
        api_key = os.environ.get(api_key_env)
        if not api_key and base_url is None:
            raise MissingAPIKeyError(
                f"Environment variable {api_key_env} is not set. Copy .env.example to .env "
                "and add your key (never commit .env)."
            )
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError("The `openai` package is not installed: pip install -r requirements.txt") from e
        self._openai = openai
        self.client = openai.OpenAI(api_key=api_key or "not-needed", base_url=base_url,
                                    timeout=timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        o = self._openai
        retryable = (o.RateLimitError, o.APITimeoutError, o.APIConnectionError, o.InternalServerError)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.time()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except o.RateLimitError as e:
                if _is_quota_error(e):
                    raise QuotaExceededError(
                        "Your OpenAI API account has no credits left (HTTP 429 insufficient_quota). "
                        "Add credits at https://platform.openai.com/settings/organization/billing "
                        "and re-run the same command."
                    ) from e
                last_err = e
                wait = min(2 ** attempt, 60)
                log.warning("Rate limited, retry %d/%d in %ds", attempt + 1, self.max_retries, wait)
                time.sleep(wait)
                continue
            except retryable as e:
                last_err = e
                wait = min(2 ** attempt, 60)
                log.warning("API error (%s), retry %d/%d in %ds", type(e).__name__,
                            attempt + 1, self.max_retries, wait)
                time.sleep(wait)
                continue
            except o.AuthenticationError as e:
                raise MissingAPIKeyError(f"Authentication failed — check your API key: {e}") from e
            except o.APIStatusError as e:  # 400 (bad request), 404 (unknown model), ...
                raise LLMError(f"API request rejected ({e.status_code}): {e}") from e

            choice = resp.choices[0] if resp.choices else None
            text = (choice.message.content or "").strip() if choice else ""
            if not text:
                raise LLMError(f"Empty response from model (finish_reason={getattr(choice, 'finish_reason', None)})")
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
            return LLMResponse(
                text=text, model=resp.model, finish_reason=choice.finish_reason,
                usage={k: v for k, v in usage.items() if isinstance(v, int)},
                system_fingerprint=getattr(resp, "system_fingerprint", None),
                latency_seconds=round(time.time() - start, 3),
            )
        raise LLMError(f"Request failed after {self.max_retries} retries: {last_err}")


class MockChatClient:
    """Offline, deterministic fake LLM (for tests / pipeline dry runs only)."""

    def __init__(self, model: str = "mock", **_: Any):
        self.model = model

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        h = int(hashlib.sha256((system + user).encode()).hexdigest(), 16)
        if json_mode and "Retrieved passages" in user:
            label = ["RELEVANT", "PARTIALLY_RELEVANT", "IRRELEVANT"][h % 3]
            text = json.dumps({"label": label, "relevant_ranks": [1] if label == "RELEVANT" else [],
                               "explanation": "mock"})
        elif json_mode:
            label = ["CORRECT", "INCORRECT", "CORRECT", "PARTIAL", "NO_ANSWER", "UNCERTAIN"][h % 6]
            text = json.dumps({"label": label, "confidence": "low", "explanation": "mock judgement"})
        else:
            text = f"[MOCK ANSWER {h % 10000}]"
        return LLMResponse(text=text, model=f"mock:{self.model}", finish_reason="stop")


def make_client(section: dict[str, Any], seed: int | None):
    """Create a client from the `llm` or `evaluation` config section."""
    if section["provider"] == "mock":
        return MockChatClient(model=section["model"])
    if section["provider"] == "openai":
        return OpenAIChatClient(
            model=section["model"], temperature=float(section["temperature"]),
            max_tokens=int(section["max_tokens"]), seed=seed,
            api_key_env=section.get("api_key_env", "OPENAI_API_KEY"),
            base_url=section.get("base_url"),
            timeout=section.get("request_timeout_seconds", 60),
            max_retries=section.get("max_retries", 5),
        )
    raise ValueError(f"Unknown provider: {section['provider']}")
