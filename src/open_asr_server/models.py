"""Pydantic models for OpenAI-compatible API requests and responses."""

from typing import Literal

from pydantic import BaseModel, Field

ResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]
DeviceType = Literal["cpu", "metal", "cuda", "remote"]
PrecisionType = Literal["fp32", "fp16", "bf16", "int8", "int4"]


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


class ModelCapabilitiesResponse(BaseModel):
    """Capability flags for a model backend."""

    supports_prompt: bool | None = None
    supports_word_timestamps: bool | None = None
    supports_segments: bool | None = None
    supports_languages: list[str] | None = None
    supports_streaming: bool | None = None


class ModelMetadataEntry(BaseModel):
    """Metadata for /v1/models/metadata endpoint."""

    id: str
    backend: str
    family: str | None = None
    parameters: int | None = None
    size_on_disk_mb: float | None = None
    weights_mb: float | None = None
    precision: PrecisionType | None = None
    device_types: list[DeviceType] | None = None
    min_ram_mb: float | None = None
    min_vram_mb: float | None = None
    capabilities: ModelCapabilitiesResponse | None = None
    notes: str | None = None
    source: Literal["model-card", "heuristic", "default", "unknown"] | None = None


class ModelMetadataListResponse(BaseModel):
    """Response for /v1/models/metadata endpoint."""

    object: str = "list"
    data: list[ModelMetadataEntry]


class ModelUnloadRequest(BaseModel):
    """Request payload for unloading a model instance."""

    model: str
    force: bool = False


class ModelUnloadAllRequest(BaseModel):
    """Request payload for unloading all model instances."""

    include_pinned: bool = False


class ModelUnloadResponse(BaseModel):
    """Response payload for model unload operations."""

    unloaded: list[str]
    skipped: list[str] = Field(default_factory=list)
    loaded: list[str]


class ModelStatusEntry(BaseModel):
    """Status for a loaded backend model instance."""

    id: str
    backend: str
    model: str
    pinned: bool
    active_requests: int
    idle_seconds: float


class ModelStatusResponse(BaseModel):
    """Response payload for model status operations."""

    data: list[ModelStatusEntry]
