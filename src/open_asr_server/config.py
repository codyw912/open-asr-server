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


@dataclass
class ServerConfig:
    """Configuration for the ASR server."""

    host: str = "127.0.0.1"
    port: int = 8000
    preload_models: list[str] = field(default_factory=list)
    default_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    max_upload_bytes: int | None = 25 * 1024 * 1024
    allowed_models: list[str] = field(default_factory=list)
    api_key: str | None = None
    rate_limit_per_minute: int | None = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        default_model = os.getenv(
            "OPEN_ASR_SERVER_DEFAULT_MODEL",
            cls().default_model,
        )
        preload_models = _parse_env_list(os.getenv("OPEN_ASR_SERVER_PRELOAD"))
        allowed_models = _parse_env_list(os.getenv("OPEN_ASR_SERVER_ALLOWED_MODELS"))
        max_upload_bytes = _parse_env_int(
            os.getenv("OPEN_ASR_SERVER_MAX_UPLOAD_BYTES"),
            cls().max_upload_bytes,
        )
        api_key = os.getenv("OPEN_ASR_SERVER_API_KEY")
        rate_limit_per_minute = _parse_env_int(
            os.getenv("OPEN_ASR_SERVER_RATE_LIMIT_PER_MINUTE"),
            cls().rate_limit_per_minute,
        )
        return cls(
            preload_models=preload_models,
            default_model=default_model,
            max_upload_bytes=max_upload_bytes,
            allowed_models=allowed_models,
            api_key=api_key,
            rate_limit_per_minute=rate_limit_per_minute,
        )
