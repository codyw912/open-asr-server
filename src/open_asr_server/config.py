"""Server configuration."""

import os
from dataclasses import dataclass, field


def _parse_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_env_int(value: str | None, default: int | None) -> int | None:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_env_float(value: str | None, default: float | None) -> float | None:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_env_bool(value: str | None, default: bool | None) -> bool | None:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class ServerConfig:
    """Configuration for the ASR server."""

    host: str = "127.0.0.1"
    port: int = 8000
    preload_models: list[str] = field(default_factory=list)
    default_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    default_backend: str | None = None
    max_upload_bytes: int | None = 25 * 1024 * 1024
    allowed_models: list[str] = field(default_factory=list)
    api_key: str | None = None
    rate_limit_per_minute: int | None = None
    transcribe_timeout_seconds: float | None = None
    transcribe_workers: int | None = None
    model_idle_seconds: float | None = None
    model_evict_interval_seconds: float | None = 60.0
    evict_preloaded_models: bool = False

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        default_model = os.getenv(
            "OPEN_ASR_SERVER_DEFAULT_MODEL",
            cls().default_model,
        )
        preload_models = _parse_env_list(os.getenv("OPEN_ASR_SERVER_PRELOAD"))
        allowed_models = _parse_env_list(os.getenv("OPEN_ASR_SERVER_ALLOWED_MODELS"))
        default_backend = os.getenv("OPEN_ASR_DEFAULT_BACKEND")
        if default_backend:
            default_backend = default_backend.strip()
        max_upload_bytes = _parse_env_int(
            os.getenv("OPEN_ASR_SERVER_MAX_UPLOAD_BYTES"),
            cls().max_upload_bytes,
        )
        api_key = os.getenv("OPEN_ASR_SERVER_API_KEY")
        rate_limit_per_minute = _parse_env_int(
            os.getenv("OPEN_ASR_SERVER_RATE_LIMIT_PER_MINUTE"),
            cls().rate_limit_per_minute,
        )
        transcribe_timeout_seconds = _parse_env_float(
            os.getenv("OPEN_ASR_SERVER_TRANSCRIBE_TIMEOUT_SECONDS"),
            cls().transcribe_timeout_seconds,
        )
        transcribe_workers = _parse_env_int(
            os.getenv("OPEN_ASR_SERVER_TRANSCRIBE_WORKERS"),
            cls().transcribe_workers,
        )
        model_idle_seconds = _parse_env_float(
            os.getenv("OPEN_ASR_SERVER_MODEL_IDLE_SECONDS"),
            cls().model_idle_seconds,
        )
        model_evict_interval_seconds = _parse_env_float(
            os.getenv("OPEN_ASR_SERVER_MODEL_EVICT_INTERVAL_SECONDS"),
            cls().model_evict_interval_seconds,
        )
        evict_preloaded_models = _parse_env_bool(
            os.getenv("OPEN_ASR_SERVER_EVICT_PRELOADED_MODELS"),
            cls().evict_preloaded_models,
        )
        return cls(
            preload_models=preload_models,
            default_model=default_model,
            default_backend=default_backend,
            max_upload_bytes=max_upload_bytes,
            allowed_models=allowed_models,
            api_key=api_key,
            rate_limit_per_minute=rate_limit_per_minute,
            transcribe_timeout_seconds=transcribe_timeout_seconds,
            transcribe_workers=transcribe_workers,
            model_idle_seconds=model_idle_seconds,
            model_evict_interval_seconds=model_evict_interval_seconds,
            evict_preloaded_models=bool(evict_preloaded_models),
        )
