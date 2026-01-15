"""Server configuration."""

from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Configuration for the ASR server."""

    host: str = "127.0.0.1"
    port: int = 8000
    preload_models: list[str] = field(default_factory=list)
    default_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
