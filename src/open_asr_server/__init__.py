"""OpenAI-compatible ASR server for local transcription."""

from importlib.metadata import PackageNotFoundError, version

from .config import ServerConfig


def _resolve_version() -> str:
    try:
        return version("open-asr-server")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()


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
