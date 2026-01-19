"""FastAPI application factory."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .backends import preload_backend
from .config import ServerConfig
from .routes import router
from .utils.rate_limit import RateLimiter


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Server configuration. Uses defaults if not provided.

    Returns:
        Configured FastAPI application.
    """
    config = config or ServerConfig.from_env()

    transcribe_executor = (
        ThreadPoolExecutor(max_workers=config.transcribe_workers)
        if config.transcribe_workers
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: preload models if configured
        for model in config.preload_models:
            preload_backend(model)
        yield
        # Shutdown: close executor if created
        if transcribe_executor:
            transcribe_executor.shutdown(wait=False)

    app = FastAPI(
        title="OpenAI-Compatible ASR Server",
        description="Local transcription server with OpenAI Whisper API compatibility",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.config = config
    app.state.rate_limiter = (
        RateLimiter(config.rate_limit_per_minute)
        if config.rate_limit_per_minute
        else None
    )
    app.state.transcribe_executor = transcribe_executor
    app.include_router(router)

    return app
