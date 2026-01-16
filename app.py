"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .backends import preload_backend
from .config import ServerConfig
from .routes import router


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Server configuration. Uses defaults if not provided.

    Returns:
        Configured FastAPI application.
    """
    config = config or ServerConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: preload models if configured
        for model in config.preload_models:
            preload_backend(model)
        yield
        # Shutdown: nothing to clean up currently

    app = FastAPI(
        title="OpenAI-Compatible ASR Server",
        description="Local transcription server with OpenAI Whisper API compatibility",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    return app
