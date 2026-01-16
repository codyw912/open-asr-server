"""Server configuration."""

import os
from dataclasses import dataclass, field


def _parse_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class ServerConfig:
    """Configuration for the ASR server."""

    host: str = "127.0.0.1"
    port: int = 8000
    preload_models: list[str] = field(default_factory=list)
    default_model: str = "mlx-community/parakeet-tdt-0.6b-v3"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        default_model = os.getenv(
            "OPENAI_ASR_SERVER_DEFAULT_MODEL",
            cls().default_model,
        )
        preload_models = _parse_env_list(os.getenv("OPENAI_ASR_SERVER_PRELOAD"))
        return cls(
            preload_models=preload_models,
            default_model=default_model,
        )
