"""Faster-Whisper transcription backend (CPU)."""

from __future__ import annotations

from pathlib import Path

from .base import Segment, TranscriptionResult, WordSegment


class FasterWhisperBackend:
    """CPU transcription backend powered by faster-whisper."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        compute_type: str = "int8",
        device: str = "cpu",
        beam_size: int = 5,
        batch_size: int = 1,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_id = model_id
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file using faster-whisper."""
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            temperature=temperature,
            beam_size=self.beam_size,
            word_timestamps=word_timestamps,
            initial_prompt=prompt,
            batch_size=self.batch_size,
        )

        words: list[WordSegment] | None = None
        segments_out: list[Segment] | None = None
        text_parts: list[str] = []

        for index, segment in enumerate(segments):
            text = segment.text.strip()
            text_parts.append(text)
            segments_out = segments_out or []
            segments_out.append(
                Segment(
                    id=index,
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    confidence=None,
                )
            )

            if word_timestamps and segment.words:
                words = words or []
                for word in segment.words:
                    words.append(
                        WordSegment(
                            word=word.word,
                            start=word.start,
                            end=word.end,
                        )
                    )

        duration = segments_out[-1].end if segments_out else 0.0
        full_text = " ".join(text_parts).strip()

        return TranscriptionResult(
            text=full_text,
            language=getattr(info, "language", None),
            duration=duration,
            words=words,
            segments=segments_out,
        )

    @property
    def supported_languages(self) -> list[str] | None:
        """Supports Whisper language auto-detection."""
        return None


def _create_faster_whisper_backend(model_id: str) -> FasterWhisperBackend:
    """Factory function for creating faster-whisper backends."""
    return FasterWhisperBackend(model_id=model_id)


# Register backends on module import
from . import BackendCapabilities, BackendDescriptor, register_backend

FASTER_WHISPER_DESCRIPTOR = BackendDescriptor(
    id="faster-whisper",
    display_name="Faster Whisper",
    model_patterns=["openai/whisper-*", "distil-whisper/*"],
    device_types=["cpu"],
    optional_dependencies=["faster-whisper"],
    capabilities=BackendCapabilities(
        supports_prompt=True,
        supports_word_timestamps=True,
        supports_segments=True,
        supports_languages=None,
    ),
    metadata={
        "family": "whisper",
        "source": "default",
    },
)

register_backend(FASTER_WHISPER_DESCRIPTOR, _create_faster_whisper_backend)
