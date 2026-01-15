"""API route handlers for OpenAI-compatible transcription endpoint."""

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .backends import get_backend, list_loaded_models, list_registered_patterns
from .formatters import to_json, to_srt, to_text, to_verbose_json, to_vtt
from .models import ModelInfo, ModelListResponse, ResponseFormat

router = APIRouter()


@router.post("/v1/audio/transcriptions")
async def create_transcription(
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
    # Get backend for model
    backend = get_backend(model)
    if not backend:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found. Available patterns: {list_registered_patterns()}",
        )

    # Save upload to temp file
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Transcribe
        word_timestamps = bool(
            timestamp_granularities and "word" in timestamp_granularities
        )
        result = backend.transcribe(
            tmp_path,
            language=language,
            temperature=temperature,
            word_timestamps=word_timestamps,
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
async def list_models():
    """List available models.

    Returns registered model patterns and currently loaded model instances.
    """
    # Combine registered patterns and loaded models
    patterns = list_registered_patterns()
    loaded = list_loaded_models()

    # Create model info for each unique entry
    all_models = set(patterns + loaded)
    data = [ModelInfo(id=m) for m in sorted(all_models)]

    return ModelListResponse(data=data)
