"""Backend registry for transcription engines."""

import fnmatch
from typing import Callable

from .base import TranscriptionBackend

_backends: dict[str, TranscriptionBackend] = {}
_backend_factories: dict[str, Callable[[str], TranscriptionBackend]] = {}


def register_backend(
    model_pattern: str, factory: Callable[[str], TranscriptionBackend]
) -> None:
    """Register a backend factory for a model pattern.

    Args:
        model_pattern: Glob pattern to match model names (e.g., "parakeet-*").
        factory: Callable that takes a model ID and returns a backend instance.
    """
    _backend_factories[model_pattern] = factory


def get_backend(model: str) -> TranscriptionBackend | None:
    """Get or create backend for model (lazy loading).

    Args:
        model: Model identifier to look up.

    Returns:
        Backend instance, or None if no matching factory found.
    """
    if model not in _backends:
        for pattern, factory in _backend_factories.items():
            if fnmatch.fnmatch(model, pattern) or model == pattern:
                _backends[model] = factory(model)
                break
    return _backends.get(model)


def preload_backend(model: str) -> TranscriptionBackend | None:
    """Eagerly load a backend.

    Args:
        model: Model identifier to preload.

    Returns:
        Backend instance, or None if no matching factory found.
    """
    return get_backend(model)


def list_registered_patterns() -> list[str]:
    """List registered model patterns."""
    return list(_backend_factories.keys())


def list_loaded_models() -> list[str]:
    """List currently loaded model instances."""
    return list(_backends.keys())


# Import backends to trigger registration
from . import parakeet as _parakeet  # noqa: F401, E402
