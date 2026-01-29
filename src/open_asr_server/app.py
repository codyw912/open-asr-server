"""FastAPI application factory."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from .backends import evict_idle_backends, preload_backend
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
        eviction_task = None
        stop_event = asyncio.Event()
        # Startup: preload models if configured
        for model in config.preload_models:
            preload_backend(
                model,
                default_backend=config.default_backend,
                pinned=not config.evict_preloaded_models,
            )
        idle_seconds = config.model_idle_seconds
        if idle_seconds is not None and idle_seconds > 0:
            interval = config.model_evict_interval_seconds or 60.0

            async def _eviction_loop():
                while not stop_event.is_set():
                    await asyncio.sleep(interval)
                    evict_idle_backends(
                        idle_seconds,
                        include_pinned=config.evict_preloaded_models,
                    )

            eviction_task = asyncio.create_task(_eviction_loop())
        yield
        # Shutdown: close executor if created
        if eviction_task:
            stop_event.set()
            eviction_task.cancel()
            with suppress(asyncio.CancelledError):
                await eviction_task
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
