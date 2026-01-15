"""Lightning Whisper MLX transcription backend.

Lightning Whisper MLX is an optimized Whisper implementation claiming
10x faster than Whisper.cpp and 4x faster than standard MLX Whisper.

Note: This backend requires lightning-whisper-mlx which has a dependency
on tiktoken==0.3.3. This may not build on Python 3.14+. The backend
will only register if the package is successfully importable.
"""

from pathlib import Path

from .base import Segment, TranscriptionResult, WordSegment

# Model name mapping - lightning-whisper uses short names
LIGHTNING_WHISPER_MODELS = {
    "lightning-whisper-tiny": "tiny",
    "lightning-whisper-small": "small",
    "lightning-whisper-base": "base",
    "lightning-whisper-medium": "medium",
    "lightning-whisper-large": "large",
    "lightning-whisper-large-v2": "large-v2",
    "lightning-whisper-large-v3": "large-v3",
    # Distilled models (faster)
    "lightning-whisper-distil-small.en": "distil-small.en",
    "lightning-whisper-distil-medium.en": "distil-medium.en",
    "lightning-whisper-distil-large-v2": "distil-large-v2",
    "lightning-whisper-distil-large-v3": "distil-large-v3",
}


class LightningWhisperBackend:
    """Lightning Whisper MLX transcription backend.

    Optimized Whisper implementation with batched decoding,
    distilled models, and quantization support.
    """

    def __init__(
        self,
        model_id: str = "lightning-whisper-distil-large-v3",
        batch_size: int = 12,
        quantization: str | None = None,
    ):
        from lightning_whisper_mlx import LightningWhisperMLX

        # Resolve model name
        model_name = LIGHTNING_WHISPER_MODELS.get(model_id, model_id)
        # Strip prefix if someone passes the full name
        if model_name.startswith("lightning-whisper-"):
            model_name = model_name[len("lightning-whisper-") :]

        self.model_id = model_id
        self.model_name = model_name
        self.whisper = LightningWhisperMLX(
            model=model_name,
            batch_size=batch_size,
            quant=quantization,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio file using Lightning Whisper MLX.

        Note: Lightning Whisper has a simpler API than standard mlx-whisper.
        It doesn't support word-level timestamps or language hints directly.
        """
        result = self.whisper.transcribe(audio_path=str(audio_path))

        # Lightning Whisper returns a simpler result format
        text = result.get("text", "").strip()

        # Extract segments if available
        segments = None
        if result.get("segments"):
            segments = [
                Segment(
                    id=i,
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                    confidence=None,
                )
                for i, seg in enumerate(result["segments"])
            ]

        # Calculate duration
        duration = 0.0
        if segments:
            duration = segments[-1].end

        return TranscriptionResult(
            text=text,
            language=result.get("language", "en"),
            duration=duration,
            words=None,  # Lightning Whisper doesn't support word timestamps
            segments=segments,
        )

    @property
    def supported_languages(self) -> list[str] | None:
        """Lightning Whisper supports same languages as Whisper."""
        return None


def _create_lightning_whisper_backend(model_id: str) -> LightningWhisperBackend:
    """Factory function for creating Lightning Whisper backends."""
    return LightningWhisperBackend(model_id)


# Try to register backends - will silently fail if lightning-whisper-mlx
# is not installed or can't be imported (e.g., tiktoken build issues on Python 3.14)
def _register_lightning_whisper_backends():
    try:
        # Test import first
        import lightning_whisper_mlx  # noqa: F401

        from . import register_backend

        # Register all model aliases
        for alias in LIGHTNING_WHISPER_MODELS:
            register_backend(alias, _create_lightning_whisper_backend)

    except ImportError:
        # lightning-whisper-mlx not installed or can't be imported
        pass


_register_lightning_whisper_backends()
