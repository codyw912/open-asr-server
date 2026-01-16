"""Base types and protocol for transcription backends."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class WordSegment:
    """A single word with timestamps."""

    word: str
    start: float
    end: float


@dataclass
class Segment:
    """A segment (sentence/phrase) with timestamps."""

    id: int
    start: float
    end: float
    text: str
    confidence: float | None = None


@dataclass
class TranscriptionResult:
    """Unified result format for all backends."""

    text: str
    language: str | None
    duration: float
    words: list[WordSegment] | None = None
    segments: list[Segment] | None = None


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Protocol for transcription backends.

    Implement this protocol to add support for a new ASR engine.
    """

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file and return unified result.

        Args:
            audio_path: Path to the audio file.
            language: Optional ISO-639-1 language code hint.
            temperature: Sampling temperature (0.0-1.0).
            word_timestamps: Whether to include word-level timestamps.
            prompt: Optional prompt to guide decoding, if supported.

        Returns:
            TranscriptionResult with text and optional timestamps.
        """
        ...

    @property
    def supported_languages(self) -> list[str] | None:
        """List of supported language codes, or None if auto-detect only."""
        ...
