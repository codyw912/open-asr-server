"""Parakeet-MLX transcription backend."""

from pathlib import Path

from .base import Segment, TranscriptionResult, WordSegment


class ParakeetBackend:
    """Parakeet-MLX transcription backend.

    Uses NVIDIA's Parakeet models via MLX for fast local transcription.
    Currently supports English only.
    """

    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v3"):
        from parakeet_mlx import from_pretrained

        self.model_id = model_id
        self.model = from_pretrained(model_id)

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio file using Parakeet-MLX.

        Note: Parakeet doesn't use language or temperature parameters,
        but they're accepted for API compatibility.
        """
        result = self.model.transcribe(str(audio_path))

        # Flatten all tokens from sentences for word-level timestamps
        words = None
        if word_timestamps and result.sentences:
            words = []
            for sentence in result.sentences:
                for token in sentence.tokens:
                    words.append(
                        WordSegment(
                            word=token.text,
                            start=token.start,
                            end=token.end,
                        )
                    )

        # Convert sentences to segments
        segments = None
        if result.sentences:
            segments = [
                Segment(
                    id=i,
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    confidence=s.confidence,
                )
                for i, s in enumerate(result.sentences)
            ]

        duration = result.sentences[-1].end if result.sentences else 0.0

        return TranscriptionResult(
            text=result.text,
            language="en",  # Parakeet is English-only
            duration=duration,
            words=words,
            segments=segments,
        )

    @property
    def supported_languages(self) -> list[str]:
        """Parakeet currently only supports English."""
        return ["en"]


def _create_parakeet_backend(model_id: str) -> ParakeetBackend:
    """Factory function for creating Parakeet backends."""
    # For short aliases, use the default model
    if not model_id.startswith("mlx-community/"):
        return ParakeetBackend()
    return ParakeetBackend(model_id)


# Register backends on module import
from . import register_backend

register_backend("parakeet-*", _create_parakeet_backend)
register_backend("mlx-community/parakeet-*", _create_parakeet_backend)
