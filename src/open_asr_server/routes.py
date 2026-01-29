"""API route handlers for OpenAI-compatible transcription endpoint."""

import asyncio
import fnmatch
import functools
import inspect
import secrets
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from .backends import (
    BackendConflictError,
    BackendNotFoundError,
    backend_use,
    list_backend_descriptors,
    list_backend_status_entries,
    list_loaded_models,
    list_loaded_model_specs,
    list_registered_patterns,
    unload_all_backends,
    unload_backend,
)
from .config import ServerConfig
from .formatters import to_json, to_srt, to_text, to_verbose_json, to_vtt
from .models import (
    ModelCapabilitiesResponse,
    ModelInfo,
    ModelListResponse,
    ModelMetadataEntry,
    ModelMetadataListResponse,
    ModelUnloadAllRequest,
    ModelUnloadRequest,
    ModelUnloadResponse,
    ModelStatusEntry,
    ModelStatusResponse,
    ResponseFormat,
)

router = APIRouter()

_CHUNK_SIZE = 1024 * 1024


def _matches_any(model: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(model, pattern) for pattern in patterns)


def _matches_allowed(model: str, patterns: list[str]) -> bool:
    if _matches_any(model, patterns):
        return True
    prefix, sep, remainder = model.partition(":")
    if sep and _matches_any(remainder, patterns):
        return True
    return False


def _extract_api_key(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization")
    if auth_header:
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
    return request.headers.get("X-API-Key")


def _ensure_authorized(request: Request, api_key: str | None) -> None:
    if not api_key:
        return
    provided = _extract_api_key(request)
    if not provided or not secrets.compare_digest(provided, api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _ensure_model_allowed(model: str, allowed: list[str]) -> None:
    if allowed and not _matches_allowed(model, allowed):
        raise HTTPException(status_code=403, detail="Model not allowed")


def _rate_limit_key(request: Request, api_key: str | None) -> str:
    if api_key:
        return _extract_api_key(request) or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _ensure_rate_limit(request: Request, api_key: str | None) -> None:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if not limiter:
        return
    key = _rate_limit_key(request, api_key)
    if not limiter.allow(key):
        raise HTTPException(status_code=429, detail="Too many requests")


def _descriptor_to_metadata(descriptor, model_id: str) -> ModelMetadataEntry:
    metadata = descriptor.metadata or {}
    entry = {
        "id": model_id,
        "backend": descriptor.id,
        "device_types": descriptor.device_types,
    }
    if descriptor.capabilities:
        entry["capabilities"] = ModelCapabilitiesResponse(
            **descriptor.capabilities.model_dump(exclude_none=True)
        )

    for key in (
        "family",
        "parameters",
        "size_on_disk_mb",
        "weights_mb",
        "precision",
        "min_ram_mb",
        "min_vram_mb",
        "notes",
        "source",
    ):
        if key in metadata:
            entry[key] = metadata[key]

    return ModelMetadataEntry(**entry)


async def _run_transcription(
    backend,
    audio_path: Path,
    transcribe_kwargs: dict,
    timeout_seconds: float | None,
    executor,
):
    loop = asyncio.get_running_loop()
    func = functools.partial(backend.transcribe, audio_path, **transcribe_kwargs)
    task = loop.run_in_executor(executor, func)
    if timeout_seconds:
        try:
            return await asyncio.wait_for(task, timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Transcription timed out")
    return await task


async def _save_upload_to_tempfile(
    file: UploadFile, max_upload_bytes: int | None
) -> Path:
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    size = 0
    try:
        while True:
            chunk = await file.read(_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if max_upload_bytes is not None and size > max_upload_bytes:
                raise HTTPException(status_code=413, detail="File too large")
            tmp.write(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        tmp.close()
        await file.close()

    return tmp_path


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: Annotated[UploadFile, File(description="The audio file to transcribe")],
    model: Annotated[str, Form(description="Model ID to use for transcription")],
    language: Annotated[
        str | None, Form(description="ISO-639-1 language code (optional)")
    ] = None,
    prompt: Annotated[
        str | None, Form(description="Optional text to guide the model's style")
    ] = None,
    response_format: Annotated[
        ResponseFormat, Form(description="Output format")
    ] = "json",
    temperature: Annotated[
        float, Form(description="Sampling temperature (0.0-1.0)")
    ] = 0.0,
    timestamp_granularities: Annotated[
        list[str] | None,
        Form(
            alias="timestamp_granularities[]",
            description="Timestamp granularity: 'word' and/or 'segment'",
        ),
    ] = None,
):
    """Create a transcription of the provided audio file.

    This endpoint is compatible with the OpenAI Whisper API.
    """
    config: ServerConfig = request.app.state.config
    _ensure_authorized(request, config.api_key)
    _ensure_rate_limit(request, config.api_key)
    _ensure_model_allowed(model, config.allowed_models)

    try:
        with backend_use(model, default_backend=config.default_backend) as backend:
            # Save upload to temp file
            tmp_path = await _save_upload_to_tempfile(file, config.max_upload_bytes)

            try:
                # Transcribe
                word_timestamps = bool(
                    timestamp_granularities and "word" in timestamp_granularities
                )
                transcribe_kwargs = {
                    "language": language,
                    "temperature": temperature,
                    "word_timestamps": word_timestamps,
                }
                if (
                    prompt
                    and "prompt" in inspect.signature(backend.transcribe).parameters
                ):
                    transcribe_kwargs["prompt"] = prompt

                executor = getattr(request.app.state, "transcribe_executor", None)
                result = await _run_transcription(
                    backend,
                    tmp_path,
                    transcribe_kwargs,
                    config.transcribe_timeout_seconds,
                    executor,
                )

                # Format response based on requested format
                if response_format == "text":
                    return PlainTextResponse(to_text(result))
                elif response_format == "srt":
                    return PlainTextResponse(to_srt(result), media_type="text/plain")
                elif response_format == "vtt":
                    return PlainTextResponse(to_vtt(result), media_type="text/vtt")
                elif response_format == "verbose_json":
                    include_words = bool(
                        timestamp_granularities and "word" in timestamp_granularities
                    )
                    include_segments = (
                        not timestamp_granularities
                        or "segment" in timestamp_granularities
                    )
                    return to_verbose_json(
                        result,
                        include_words=include_words,
                        include_segments=include_segments,
                    )
                else:  # json (default)
                    return to_json(result)

            finally:
                tmp_path.unlink(missing_ok=True)
    except BackendNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{exc.model}' not found. Available patterns: {exc.patterns}"
            ),
        )
    except BackendConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Model '"
                + exc.model
                + "' matches multiple backends: "
                + ", ".join(sorted(exc.candidates))
                + ". Use backend:model or set OPEN_ASR_DEFAULT_BACKEND."
            ),
        )


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request):
    """List available models.

    Returns registered model patterns and currently loaded model instances.
    """
    config: ServerConfig = request.app.state.config
    _ensure_authorized(request, config.api_key)
    _ensure_rate_limit(request, config.api_key)

    # Combine registered patterns and loaded models
    patterns = list_registered_patterns()
    loaded = list_loaded_models()

    if config.allowed_models:
        patterns = [
            pattern
            for pattern in patterns
            if _matches_allowed(pattern, config.allowed_models)
        ]
        loaded = [
            model for model in loaded if _matches_allowed(model, config.allowed_models)
        ]

    # Create model info for each unique entry
    all_models = set(patterns + loaded)
    data = [ModelInfo(id=m) for m in sorted(all_models)]

    return ModelListResponse(data=data)


@router.get(
    "/v1/models/metadata",
    response_model=ModelMetadataListResponse,
    response_model_exclude_none=True,
)
async def list_models_metadata(request: Request):
    """List model metadata entries."""
    config: ServerConfig = request.app.state.config
    _ensure_authorized(request, config.api_key)
    _ensure_rate_limit(request, config.api_key)

    descriptors = {
        descriptor.id: descriptor for descriptor in list_backend_descriptors()
    }
    entries: dict[tuple[str, str], ModelMetadataEntry] = {}

    for descriptor in descriptors.values():
        for pattern in descriptor.model_patterns:
            entry_id = pattern
            if config.allowed_models and not _matches_allowed(
                entry_id, config.allowed_models
            ):
                continue
            entries[(descriptor.id, entry_id)] = _descriptor_to_metadata(
                descriptor, entry_id
            )

    for backend_id, model_id in list_loaded_model_specs():
        descriptor = descriptors.get(backend_id)
        if not descriptor:
            continue
        entry_id = f"{backend_id}:{model_id}"
        if config.allowed_models and not _matches_allowed(
            entry_id, config.allowed_models
        ):
            continue
        entries[(backend_id, entry_id)] = _descriptor_to_metadata(descriptor, entry_id)

    data = [entries[key] for key in sorted(entries.keys())]
    return ModelMetadataListResponse(data=data)


@router.post("/v1/admin/models/unload", response_model=ModelUnloadResponse)
async def unload_model(request: Request, payload: ModelUnloadRequest):
    """Unload a single backend model instance."""
    config: ServerConfig = request.app.state.config
    _ensure_authorized(request, config.api_key)
    _ensure_rate_limit(request, config.api_key)

    try:
        result = unload_backend(
            payload.model,
            default_backend=config.default_backend,
            include_pinned=payload.force,
        )
    except BackendNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{exc.model}' not found. Available patterns: {exc.patterns}"
            ),
        )
    except BackendConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Model '"
                + exc.model
                + "' matches multiple backends: "
                + ", ".join(sorted(exc.candidates))
                + ". Use backend:model or set OPEN_ASR_DEFAULT_BACKEND."
            ),
        )

    if result.status == "not_loaded":
        raise HTTPException(status_code=404, detail="Model is not loaded")
    if result.status == "pinned":
        raise HTTPException(
            status_code=409,
            detail="Model is pinned; set force=true to unload",
        )
    if result.status == "in_use":
        raise HTTPException(status_code=409, detail="Model is in use")

    return ModelUnloadResponse(
        unloaded=[result.model] if result.model else [],
        skipped=[],
        loaded=list_loaded_models(),
    )


@router.post("/v1/admin/models/unload-all", response_model=ModelUnloadResponse)
async def unload_all_models(request: Request, payload: ModelUnloadAllRequest):
    """Unload all cached backend model instances."""
    config: ServerConfig = request.app.state.config
    _ensure_authorized(request, config.api_key)
    _ensure_rate_limit(request, config.api_key)

    unloaded, skipped = unload_all_backends(
        include_pinned=payload.include_pinned,
    )
    return ModelUnloadResponse(
        unloaded=unloaded,
        skipped=skipped,
        loaded=list_loaded_models(),
    )


@router.get("/v1/admin/models/status", response_model=ModelStatusResponse)
async def model_status(request: Request):
    """Get status for loaded backend model instances."""
    config: ServerConfig = request.app.state.config
    _ensure_authorized(request, config.api_key)
    _ensure_rate_limit(request, config.api_key)

    data = [
        ModelStatusEntry(
            id=f"{entry.backend_id}:{entry.model_id}",
            backend=entry.backend_id,
            model=entry.model_id,
            pinned=entry.pinned,
            active_requests=entry.active_requests,
            idle_seconds=entry.idle_seconds,
        )
        for entry in list_backend_status_entries()
    ]
    return ModelStatusResponse(data=data)
