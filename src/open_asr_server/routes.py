"""API route handlers for OpenAI-compatible transcription endpoint."""

import fnmatch
import inspect
import secrets
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from .backends import get_backend, list_loaded_models, list_registered_patterns
from .config import ServerConfig
from .formatters import to_json, to_srt, to_text, to_verbose_json, to_vtt
from .models import ModelInfo, ModelListResponse, ResponseFormat

router = APIRouter()

_CHUNK_SIZE = 1024 * 1024


def _matches_any(model: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(model, pattern) for pattern in patterns)


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
    if allowed and not _matches_any(model, allowed):
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

    # Get backend for model
    backend = get_backend(model)
    if not backend:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found. Available patterns: {list_registered_patterns()}",
        )

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
        if prompt and "prompt" in inspect.signature(backend.transcribe).parameters:
            transcribe_kwargs["prompt"] = prompt

        result = backend.transcribe(tmp_path, **transcribe_kwargs)

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
                not timestamp_granularities or "segment" in timestamp_granularities
            )
            return to_verbose_json(
                result, include_words=include_words, include_segments=include_segments
            )
        else:  # json (default)
            return to_json(result)

    finally:
        tmp_path.unlink(missing_ok=True)


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
            if _matches_any(pattern, config.allowed_models)
        ]
        loaded = [
            model for model in loaded if _matches_any(model, config.allowed_models)
        ]

    # Create model info for each unique entry
    all_models = set(patterns + loaded)
    data = [ModelInfo(id=m) for m in sorted(all_models)]

    return ModelListResponse(data=data)
