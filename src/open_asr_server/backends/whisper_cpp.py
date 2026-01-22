"""whisper.cpp transcription backend (pywhispercpp)."""

from __future__ import annotations

from pathlib import Path

from .base import Segment, TranscriptionResult, WordSegment

WHISPER_CPP_PATTERNS = [
    "tiny*",
    "base*",
    "small*",
    "medium*",
    "large*",
]


class WhisperCppBackend:
    """CPU-first backend powered by whisper.cpp bindings."""

    def __init__(self, model_id: str = "base.en") -> None:
        from pywhispercpp.model import Model

        self.model_id = model_id
        self.model = Model(
            model_id,
            print_progress=False,
            print_realtime=False,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file using whisper.cpp."""
        if word_timestamps:
            word_timestamps = False

        params = {
            "temperature": temperature,
        }
        if language is not None:
            params["language"] = language
        if prompt:
            params["initial_prompt"] = prompt

        segments = self.model.transcribe(str(audio_path), **params)

        segments_out: list[Segment] | None = None
        words: list[WordSegment] | None = None
        text_parts: list[str] = []

        for index, segment in enumerate(segments):
            text = str(segment.text).strip()
            text_parts.append(text)
            segments_out = segments_out or []
            segments_out.append(
                Segment(
                    id=index,
                    start=segment.t0 * 0.01,
                    end=segment.t1 * 0.01,
                    text=text,
                    confidence=(
                        None
                        if getattr(segment, "probability", None) is None
                        else float(segment.probability)
                    ),
                )
            )

        duration = segments_out[-1].end if segments_out else 0.0
        full_text = " ".join(text_parts).strip()

        return TranscriptionResult(
            text=full_text,
            language=language,
            duration=duration,
            words=words,
            segments=segments_out,
        )

    @property
    def supported_languages(self) -> list[str] | None:
        """Supports Whisper language auto-detection."""
        return None


def _create_whisper_cpp_backend(model_id: str) -> WhisperCppBackend:
    """Factory function for creating whisper.cpp backends."""
    return WhisperCppBackend(model_id=model_id)


# Register backends on module import
from . import BackendCapabilities, BackendDescriptor, register_backend

WHISPER_CPP_DESCRIPTOR = BackendDescriptor(
    id="whisper-cpp",
    display_name="Whisper.cpp",
    model_patterns=WHISPER_CPP_PATTERNS,
    device_types=["cpu"],
    optional_dependencies=["pywhispercpp"],
    capabilities=BackendCapabilities(
        supports_prompt=True,
        supports_word_timestamps=False,
        supports_segments=True,
        supports_languages=None,
    ),
    metadata={
        "family": "whisper",
        "notes": "pywhispercpp supports Metal/CoreML when built with WHISPER_COREML=1",
        "source": "default",
    },
)

register_backend(WHISPER_CPP_DESCRIPTOR, _create_whisper_cpp_backend)
