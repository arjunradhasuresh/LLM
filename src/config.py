"""Load and validate the experiment configuration (config/config.yaml).

All scripts obtain their parameters through :func:`load_config`, so every
experimental setting is documented in exactly one place.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of the `src` package directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"

REQUIRED_KEYS = {
    "experiment": ["name", "random_seed"],
    "dataset": ["name", "url", "raw_path", "processed_path", "number_of_questions"],
    "corpus": ["scope", "snapshot_timestamp", "api_url", "documents_path", "chunks_path"],
    "retrieval": ["embedding_model", "chunk_size", "chunk_overlap", "top_k", "index_dir"],
    "llm": ["provider", "model", "temperature", "max_tokens"],
    "evaluation": ["provider", "model", "temperature", "max_tokens"],
    "prompts": ["system", "no_rag", "rag", "judge_answer", "judge_context"],
    "paths": ["results_raw", "results_processed", "results_tables", "results_figures"],
}
SUPPORTED_PROVIDERS = {"openai", "mock"}
SUPPORTED_DATASETS = {"truthfulqa"}
SUPPORTED_CORPUS_SCOPES = {"all_truthfulqa", "sampled_only"}


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


def resolve_path(path: str | Path) -> Path:
    """Resolve a (possibly relative) config path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def validate_config(cfg: dict[str, Any]) -> None:
    """Check presence, types and ranges of all settings. Raises ConfigError."""
    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ConfigError(f"Missing config section: '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ConfigError(f"Missing config key: '{section}.{key}'")

    if cfg["dataset"]["name"] not in SUPPORTED_DATASETS:
        raise ConfigError(f"dataset.name must be one of {sorted(SUPPORTED_DATASETS)}")
    n = cfg["dataset"]["number_of_questions"]
    if not isinstance(n, int) or n <= 0:
        raise ConfigError("dataset.number_of_questions must be a positive integer")
    if not isinstance(cfg["experiment"]["random_seed"], int):
        raise ConfigError("experiment.random_seed must be an integer")
    if cfg["corpus"]["scope"] not in SUPPORTED_CORPUS_SCOPES:
        raise ConfigError(f"corpus.scope must be one of {sorted(SUPPORTED_CORPUS_SCOPES)}")

    r = cfg["retrieval"]
    for key in ("chunk_size", "top_k"):
        if not isinstance(r[key], int) or r[key] <= 0:
            raise ConfigError(f"retrieval.{key} must be a positive integer")
    if not isinstance(r["chunk_overlap"], int) or r["chunk_overlap"] < 0:
        raise ConfigError("retrieval.chunk_overlap must be a non-negative integer")
    if r["chunk_overlap"] >= r["chunk_size"]:
        raise ConfigError("retrieval.chunk_overlap must be smaller than retrieval.chunk_size")

    for section in ("llm", "evaluation"):
        s = cfg[section]
        if s["provider"] not in SUPPORTED_PROVIDERS:
            raise ConfigError(f"{section}.provider must be one of {sorted(SUPPORTED_PROVIDERS)}")
        t = s["temperature"]
        if not isinstance(t, (int, float)) or not 0.0 <= float(t) <= 2.0:
            raise ConfigError(f"{section}.temperature must be a number in [0, 2]")
        if not isinstance(s["max_tokens"], int) or s["max_tokens"] <= 0:
            raise ConfigError(f"{section}.max_tokens must be a positive integer")
        if not str(s["model"]).strip():
            raise ConfigError(f"{section}.model must not be empty")

    for key, path in cfg["prompts"].items():
        if not resolve_path(path).is_file():
            raise ConfigError(f"Prompt file for '{key}' not found: {path}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config, validate it and return it as a nested dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Config file is not valid YAML: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError("Config file must contain a YAML mapping")
    validate_config(cfg)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def fingerprint(obj: Any) -> str:
    """Short, stable hash of a JSON-serialisable object (used to tag runs)."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def generation_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """The settings that must be identical across both conditions."""
    llm = cfg["llm"]
    return {
        "provider": llm["provider"],
        "model": llm["model"],
        "base_url": llm.get("base_url"),
        "temperature": llm["temperature"],
        "max_tokens": llm["max_tokens"],
        "seed": cfg["experiment"]["random_seed"],
    }


def public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config copy without private/internal keys (safe to save with results)."""
    out = copy.deepcopy(cfg)
    out.pop("_config_path", None)
    return out


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser shared by all command-line entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Path to the YAML config (default: config/config.yaml)")
    return parser
