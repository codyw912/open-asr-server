"""Backend registry for transcription engines."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict, Field

from .base import TranscriptionBackend

_ENTRY_POINT_GROUP = "open_asr_server.backends"


class BackendCapabilities(BaseModel):
    """Optional capability flags for a backend."""

    model_config = ConfigDict(extra="forbid")

    supports_prompt: bool = False
    supports_word_timestamps: bool = False
    supports_segments: bool = False
    supports_languages: list[str] | None = None
    supports_streaming: bool = False


class BackendDescriptor(BaseModel):
    """Descriptor for a transcription backend."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-_]+$")
    display_name: str
    model_patterns: list[str] = Field(min_length=1)
    device_types: list[str] = Field(min_length=1)
    optional_dependencies: list[str] = Field(default_factory=list)
    priority: int = 0
    capabilities: BackendCapabilities | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class BackendEntryPoint:
    descriptor: BackendDescriptor
    factory: Callable[[str], TranscriptionBackend]


@dataclass(frozen=True)
class RegisteredBackend:
    descriptor: BackendDescriptor
    factory: Callable[[str], TranscriptionBackend]


class BackendResolutionError(Exception):
    """Base error for backend resolution issues."""


class BackendNotFoundError(BackendResolutionError):
    """No backend matched the requested model."""

    def __init__(self, model: str, patterns: list[str]):
        super().__init__(model)
        self.model = model
        self.patterns = patterns


class BackendConflictError(BackendResolutionError):
    """Multiple backends matched the requested model."""

    def __init__(self, model: str, candidates: list[str]):
        super().__init__(model)
        self.model = model
        self.candidates = candidates


_backends: dict[tuple[str, str], TranscriptionBackend] = {}
_registered_backends: dict[str, RegisteredBackend] = {}


def register_backend(
    descriptor: BackendDescriptor, factory: Callable[[str], TranscriptionBackend]
) -> None:
    """Register a backend factory.

    Args:
        descriptor: Backend descriptor with model patterns and metadata.
        factory: Callable that takes a model ID and returns a backend instance.
    """
    if descriptor.id in _registered_backends:
        raise ValueError(f"Backend '{descriptor.id}' is already registered")
    _registered_backends[descriptor.id] = RegisteredBackend(descriptor, factory)


def _default_backend_from_env() -> str | None:
    value = os.getenv("OPEN_ASR_DEFAULT_BACKEND")
    if value:
        return value.strip()
    return None


def _match_descriptors(model: str) -> list[BackendDescriptor]:
    matches = []
    for registered in _registered_backends.values():
        if any(
            fnmatch.fnmatch(model, pattern)
            for pattern in registered.descriptor.model_patterns
        ):
            matches.append(registered.descriptor)
    return matches


def _resolve_backend_id(model: str, default_backend: str | None) -> tuple[str, str]:
    prefix, sep, remainder = model.partition(":")
    if sep:
        if not remainder or prefix not in _registered_backends:
            raise BackendNotFoundError(model, list_registered_patterns())
        return prefix, remainder

    matches = _match_descriptors(model)
    if not matches:
        raise BackendNotFoundError(model, list_registered_patterns())

    if len(matches) == 1:
        return matches[0].id, model

    if default_backend and default_backend in {descriptor.id for descriptor in matches}:
        return default_backend, model

    raise BackendConflictError(model, [descriptor.id for descriptor in matches])


def get_backend(model: str, default_backend: str | None = None) -> TranscriptionBackend:
    """Get or create backend for model (lazy loading).

    Args:
        model: Model identifier to look up.
        default_backend: Backend ID to use when multiple matches exist.

    Returns:
        Backend instance.
    """
    if default_backend is None:
        default_backend = _default_backend_from_env()

    backend_id, model_id = _resolve_backend_id(model, default_backend)
    key = (backend_id, model_id)
    if key not in _backends:
        backend = _registered_backends[backend_id].factory(model_id)
        _backends[key] = backend
    return _backends[key]


def preload_backend(
    model: str, default_backend: str | None = None
) -> TranscriptionBackend:
    """Eagerly load a backend.

    Args:
        model: Model identifier to preload.
        default_backend: Backend ID to use when multiple matches exist.

    Returns:
        Backend instance.
    """
    return get_backend(model, default_backend=default_backend)


def list_backend_descriptors() -> list[BackendDescriptor]:
    """List registered backend descriptors."""
    return [backend.descriptor for backend in _registered_backends.values()]


def list_registered_patterns() -> list[str]:
    """List registered model patterns."""
    patterns: list[str] = []
    for backend in _registered_backends.values():
        patterns.extend(backend.descriptor.model_patterns)
    return patterns


def list_loaded_model_specs() -> list[tuple[str, str]]:
    """List loaded backend/model pairs."""
    return list(_backends.keys())


def list_loaded_models() -> list[str]:
    """List loaded model IDs with backend prefixes."""
    return [f"{backend_id}:{model_id}" for backend_id, model_id in _backends.keys()]


def _load_entry_point(entry_point: metadata.EntryPoint) -> None:
    try:
        value: Any = entry_point.load()
        if callable(value):
            value = value()

        descriptor: Any
        factory: Any
        if isinstance(value, BackendEntryPoint):
            descriptor = value.descriptor
            factory = value.factory
        elif isinstance(value, tuple) and len(value) == 2:
            descriptor, factory = value
        elif isinstance(value, dict) and "descriptor" in value and "factory" in value:
            descriptor = value["descriptor"]
            factory = value["factory"]
        elif hasattr(value, "descriptor") and hasattr(value, "factory"):
            descriptor = getattr(value, "descriptor")
            factory = getattr(value, "factory")
        else:
            raise TypeError("Entry point must provide descriptor and factory")

        descriptor = BackendDescriptor.model_validate(descriptor)
        typed_factory = cast(Callable[[str], TranscriptionBackend], factory)
        if not callable(typed_factory):
            raise TypeError("Backend factory must be callable")

        register_backend(descriptor, typed_factory)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load backend entry point '{entry_point.name}'"
        ) from exc


def _load_entry_points() -> None:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        plugins = entry_points.select(group=_ENTRY_POINT_GROUP)
    else:
        plugins = entry_points.get(_ENTRY_POINT_GROUP, [])

    for entry_point in plugins:
        _load_entry_point(entry_point)


# Import backends to trigger registration
from . import parakeet as _parakeet  # noqa: F401, E402
from . import whisper as _whisper  # noqa: F401, E402
from . import lightning_whisper as _lightning_whisper  # noqa: F401, E402
from . import faster_whisper as _faster_whisper  # noqa: F401, E402
from . import whisper_cpp as _whisper_cpp  # noqa: F401, E402
from . import kyutai_mlx as _kyutai_mlx  # noqa: F401, E402

_load_entry_points()
