"""OpenAI-compatible ASR server for local transcription."""

from .config import ServerConfig

__version__ = "0.1.0"

def create_app(config: ServerConfig | None = None):
    """Create the FastAPI application."""
    from .app import create_app as _create_app

    return _create_app(config)


try:
    app = create_app()
except ModuleNotFoundError as exc:
    if exc.name == "fastapi":
        app = None
    else:
        raise

__all__ = ["create_app", "ServerConfig", "app", "__version__"]
