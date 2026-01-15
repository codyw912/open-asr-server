"""MLX-Whisper transcription backend."""

from pathlib import Path

from .base import Segment, TranscriptionResult, WordSegment

# Default model mapping for short aliases
WHISPER_MODELS = {
    "whisper-tiny": "mlx-community/whisper-tiny",
    "whisper-base": "mlx-community/whisper-base",
    "whisper-small": "mlx-community/whisper-small",
    "whisper-medium": "mlx-community/whisper-medium",
    "whisper-large": "mlx-community/whisper-large",
    "whisper-large-v2": "mlx-community/whisper-large-v2",
    "whisper-large-v3": "mlx-community/whisper-large-v3",
    "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


class WhisperBackend:
    """MLX-Whisper transcription backend.

    Uses OpenAI's Whisper models via MLX for fast local transcription.
    Supports 99+ languages with automatic language detection.
    """

    def __init__(self, model_id: str = "mlx-community/whisper-large-v3-turbo"):
        # Resolve short aliases to full model IDs
        self.model_id = WHISPER_MODELS.get(model_id, model_id)

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio file using MLX-Whisper."""
        import mlx_whisper

        # Build decode options
        decode_options = {}
        if language:
            decode_options["language"] = language

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_id,
            temperature=temperature,
            word_timestamps=word_timestamps,
            verbose=False,
            **decode_options,
        )

        # Extract words if available
        words = None
        if word_timestamps:
            words = []
            for segment in result.get("segments", []):
                for w in segment.get("words", []):
                    words.append(
                        WordSegment(
                            word=w["word"],
                            start=float(w["start"]),
                            end=float(w["end"]),
                        )
                    )

        # Convert segments
        segments = None
        if result.get("segments"):
            segments = [
                Segment(
                    id=seg["id"],
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    confidence=1.0 - seg.get("no_speech_prob", 0.0),
                )
                for seg in result["segments"]
            ]

        # Calculate duration from last segment
        duration = 0.0
        if segments:
            duration = segments[-1].end

        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language"),
            duration=duration,
            words=words,
            segments=segments,
        )

    @property
    def supported_languages(self) -> list[str] | None:
        """Whisper supports 99+ languages with auto-detection."""
        # Return None to indicate auto-detection is supported
        return None


def _create_whisper_backend(model_id: str) -> WhisperBackend:
    """Factory function for creating Whisper backends."""
    return WhisperBackend(model_id)


# Register backends on module import
from . import register_backend

# Short aliases
for alias in WHISPER_MODELS:
    register_backend(alias, _create_whisper_backend)

# Full mlx-community model paths
register_backend("mlx-community/whisper-*", _create_whisper_backend)
