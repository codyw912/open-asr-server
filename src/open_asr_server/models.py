"""Pydantic models for OpenAI-compatible API requests and responses."""

from typing import Literal

from pydantic import BaseModel

ResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


class TranscriptionResponse(BaseModel):
    """Simple JSON response (response_format='json')."""

    text: str


class WordResponse(BaseModel):
    """Word-level timestamp response."""

    word: str
    start: float
    end: float


class SegmentResponse(BaseModel):
    """Segment (sentence/phrase) response.

    Includes fields for OpenAI API compatibility, though not all
    are populated by all backends.
    """

    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0


class VerboseTranscriptionResponse(BaseModel):
    """Verbose JSON response with timestamps (response_format='verbose_json')."""

    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[WordResponse] | None = None
    segments: list[SegmentResponse] | None = None


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""

    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """Response for /v1/models endpoint."""

    object: str = "list"
    data: list[ModelInfo]
