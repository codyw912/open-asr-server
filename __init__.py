"""OpenAI-compatible ASR server for local transcription."""

from .app import create_app
from .config import ServerConfig

__version__ = "0.1.0"

# Server config set by CLI, used by app factory
_server_config: ServerConfig | None = None


def _get_app():
    """Get app instance (used by uvicorn)."""
    return create_app(_server_config)


# Create app instance for uvicorn
app = _get_app()

__all__ = ["create_app", "ServerConfig", "app", "__version__"]
