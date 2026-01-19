"""Helpers for resolving model cache paths."""

from __future__ import annotations

import os
from pathlib import Path


def get_model_cache_dir() -> Path | None:
    value = os.getenv("OPEN_ASR_SERVER_MODEL_DIR")
    if not value:
        return None
    return Path(value).expanduser()


def get_hf_token() -> str | None:
    value = os.getenv("OPEN_ASR_SERVER_HF_TOKEN")
    if not value:
        return None
    return value


def resolve_model_path(model_id: str) -> Path:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return candidate

    cache_dir = get_model_cache_dir()
    cache_dir_value = str(cache_dir) if cache_dir is not None else None
    token = get_hf_token()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir_value,
            token=token,
        )
    )
