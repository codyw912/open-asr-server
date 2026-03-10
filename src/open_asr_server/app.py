"""FastAPI application factory."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
import fnmatch
import logging
import platform
import sys

from fastapi import FastAPI

from . import __version__
from .backends import evict_idle_backends, list_backend_descriptors, preload_backend
from .config import ServerConfig
from .install_hints import (
    backend_install_hint,
    backend_runtime_status,
    detect_nvidia_gpu,
    install_command,
)
from .routes import router
from .utils.rate_limit import RateLimiter

logger = logging.getLogger(__name__)


def _python_minor_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _platform_name() -> str:
    return platform.system().lower()


def _default_backend_candidates(model: str):
    descriptors = list_backend_descriptors()
    prefix, sep, remainder = model.partition(":")
    if sep:
        return [
            descriptor for descriptor in descriptors if descriptor.id == prefix
        ], remainder
    return [
        descriptor
        for descriptor in descriptors
        if any(fnmatch.fnmatch(model, pattern) for pattern in descriptor.model_patterns)
    ], model


def _select_default_backend_descriptor(
    candidates,
    default_backend: str | None,
):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if default_backend:
        for descriptor in candidates:
            if descriptor.id == default_backend:
                return descriptor
    return None


def _log_startup_compatibility_diagnostics(config: ServerConfig) -> None:
    candidates, resolved_model = _default_backend_candidates(config.default_model)
    selected = _select_default_backend_descriptor(candidates, config.default_backend)

    if selected is None:
        if not candidates:
            logger.warning(
                "Default model '%s' does not match any backend pattern",
                config.default_model,
            )
            return

        candidate_ids = ", ".join(sorted(descriptor.id for descriptor in candidates))
        if config.default_backend:
            logger.warning(
                "OPEN_ASR_DEFAULT_BACKEND='%s' does not match default model '%s' candidates (%s)",
                config.default_backend,
                config.default_model,
                candidate_ids,
            )
            return

        logger.warning(
            "Default model '%s' is ambiguous across backends (%s); set OPEN_ASR_DEFAULT_BACKEND",
            config.default_model,
            candidate_ids,
        )
        return

    hint = backend_install_hint(selected.id)
    if hint is None:
        logger.info(
            "Default backend '%s' selected for model '%s'",
            selected.id,
            config.default_model,
        )
        return

    has_nvidia, gpu_source = detect_nvidia_gpu()
    status, reason = backend_runtime_status(
        selected.id,
        platform_name=_platform_name(),
        python_version=_python_minor_version(),
        has_nvidia=has_nvidia,
    )

    if status == "ready":
        logger.info(
            "Default backend '%s' selected for model '%s' (gpu=%s via %s)",
            selected.id,
            config.default_model,
            "yes" if has_nvidia else "no",
            gpu_source,
        )
        return

    logger.warning(
        "Default backend '%s' for model '%s' is %s (%s). Suggested install: %s",
        selected.id,
        resolved_model,
        status,
        reason or "unknown compatibility issue",
        install_command(hint.extra, python=hint.python),
    )


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
        _log_startup_compatibility_diagnostics(config)
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
        version=__version__,
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
