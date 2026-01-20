"""Lightning Whisper MLX transcription backend.

Lightning Whisper MLX is an optimized Whisper implementation claiming
10x faster than Whisper.cpp and 4x faster than standard MLX Whisper.

Note: This backend requires lightning-whisper-mlx which has a dependency
on tiktoken==0.3.3. This may not build on Python 3.14+. The backend
will only register if the package is successfully importable.
"""

from pathlib import Path

from ..utils.model_cache import get_hf_token, get_model_cache_dir, resolve_model_path
from .base import Segment, TranscriptionResult, WordSegment

# Model name mapping - lightning-whisper uses short names
LIGHTNING_WHISPER_MODELS = {
    "lightning-whisper-tiny": ("mlx-community/whisper-tiny", None),
    "lightning-whisper-small": ("mlx-community/whisper-small-mlx", None),
    "lightning-whisper-base": ("mlx-community/whisper-base-mlx", None),
    "lightning-whisper-medium": ("mlx-community/whisper-medium-mlx", None),
    "lightning-whisper-large": ("mlx-community/whisper-large-mlx", None),
    "lightning-whisper-large-v2": ("mlx-community/whisper-large-v2-mlx", None),
    "lightning-whisper-large-v3": ("mlx-community/whisper-large-v3-mlx", None),
    # Distilled models (faster)
    "lightning-whisper-distil-small.en": (
        "mustafaaljadery/distil-whisper-mlx",
        "mlx_models/distil-small.en",
    ),
    "lightning-whisper-distil-medium.en": (
        "mustafaaljadery/distil-whisper-mlx",
        "mlx_models/distil-medium.en",
    ),
    "lightning-whisper-distil-large-v2": (
        "mustafaaljadery/distil-whisper-mlx",
        "mlx_models/distil-large-v2",
    ),
    "lightning-whisper-distil-large-v3": (
        "mustafaaljadery/distil-whisper-mlx",
        "mlx_models/distil-large-v3",
    ),
}


def _download_lightning_subdir(repo_id: str, subdir: str) -> Path:
    cache_dir = get_model_cache_dir()
    cache_dir_value = str(cache_dir) if cache_dir is not None else None
    token = get_hf_token()

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subdir}/config.json",
        cache_dir=cache_dir_value,
        token=token,
    )
    hf_hub_download(
        repo_id=repo_id,
        filename=f"{subdir}/weights.npz",
        cache_dir=cache_dir_value,
        token=token,
    )
    return Path(config_path).parent


def _resolve_lightning_model_path(model_id: str) -> Path:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return candidate

    repo_id, subdir = LIGHTNING_WHISPER_MODELS.get(model_id, (model_id, None))
    if subdir:
        return _download_lightning_subdir(repo_id, subdir)

    return resolve_model_path(repo_id)


class LightningWhisperBackend:
    """Lightning Whisper MLX transcription backend.

    Optimized Whisper implementation with batched decoding
    and distilled models.
    """

    def __init__(
        self,
        model_id: str = "lightning-whisper-distil-large-v3",
        batch_size: int = 12,
        quantization: str | None = None,
    ):
        if quantization is not None:
            raise ValueError(
                "Quantization selection is not supported; use an explicit HF repo ID."
            )

        self.model_id = model_id
        self.batch_size = batch_size
        self.model_path = _resolve_lightning_model_path(model_id)

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file using Lightning Whisper MLX.

        Note: Lightning Whisper has a simpler API than standard mlx-whisper.
        It doesn't support word-level timestamps or prompts directly.
        """
        from lightning_whisper_mlx.transcribe import transcribe_audio

        decode_options = {}
        if language:
            decode_options["language"] = language

        result = transcribe_audio(
            str(audio_path),
            path_or_hf_repo=str(self.model_path),
            temperature=temperature,
            word_timestamps=word_timestamps,
            batch_size=self.batch_size,
            **decode_options,
        )

        # Lightning Whisper returns a simpler result format
        text = result.get("text", "").strip()

        # Extract segments if available
        segments = None
        if result.get("segments"):
            segments = []
            for i, seg in enumerate(result["segments"]):
                if isinstance(seg, dict):
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", 0.0))
                    text_value = seg.get("text", "")
                elif isinstance(seg, (list, tuple)):
                    # lightning-whisper-mlx returns [start_frame, end_frame, text]
                    start_frame = seg[0] if len(seg) > 0 else 0.0
                    end_frame = seg[1] if len(seg) > 1 else start_frame
                    text_value = seg[2] if len(seg) > 2 else ""
                    try:
                        from lightning_whisper_mlx.audio import HOP_LENGTH, SAMPLE_RATE

                        start = float(start_frame) * HOP_LENGTH / SAMPLE_RATE
                        end = float(end_frame) * HOP_LENGTH / SAMPLE_RATE
                    except Exception:
                        start = float(start_frame)
                        end = float(end_frame)
                else:
                    continue

                segments.append(
                    Segment(
                        id=i,
                        start=start,
                        end=end,
                        text=str(text_value).strip(),
                        confidence=None,
                    )
                )

        # Calculate duration
        duration = 0.0
        if segments:
            duration = segments[-1].end

        language_value = result.get("language")
        if isinstance(language_value, str) and not language_value:
            language_value = None

        return TranscriptionResult(
            text=text,
            language=language_value,
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

        from . import BackendCapabilities, BackendDescriptor, register_backend

        descriptor = BackendDescriptor(
            id="lightning-whisper-mlx",
            display_name="Lightning Whisper MLX",
            model_patterns=list(LIGHTNING_WHISPER_MODELS.keys()),
            device_types=["metal"],
            optional_dependencies=["lightning-whisper-mlx"],
            capabilities=BackendCapabilities(
                supports_prompt=False,
                supports_word_timestamps=False,
                supports_segments=True,
                supports_languages=None,
            ),
            metadata={
                "family": "whisper",
                "source": "default",
            },
        )

        register_backend(descriptor, _create_lightning_whisper_backend)

    except ImportError:
        # lightning-whisper-mlx not installed or can't be imported
        pass


_register_lightning_whisper_backends()
