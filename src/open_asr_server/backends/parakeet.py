"""Parakeet-MLX transcription backend."""

from pathlib import Path

from ..utils.model_cache import get_hf_token, get_model_cache_dir
from .base import Segment, TranscriptionResult, WordSegment


def _load_parakeet_model(model_id: str):
    from parakeet_mlx import from_pretrained

    cache_dir = get_model_cache_dir()
    token = get_hf_token()

    candidate = Path(model_id).expanduser()
    if candidate.exists() or token is None:
        return from_pretrained(model_id, cache_dir=cache_dir)

    cache_dir_value = str(cache_dir) if cache_dir is not None else None

    from huggingface_hub import hf_hub_download
    from mlx.utils import tree_flatten, tree_unflatten
    import mlx.core as mx
    from parakeet_mlx.utils import from_config
    import json

    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        cache_dir=cache_dir_value,
        token=token,
    )
    weights_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors",
        cache_dir=cache_dir_value,
        token=token,
    )

    with open(config_path, "r") as handle:
        config = json.load(handle)

    model = from_config(config)
    model.load_weights(weights_path)

    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(mx.bfloat16)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    return model


class ParakeetBackend:
    """Parakeet-MLX transcription backend.

    Uses NVIDIA's Parakeet models via MLX for fast local transcription.
    Currently supports English only.
    """

    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v3"):
        self.model_id = model_id
        self.model = _load_parakeet_model(model_id)

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file using Parakeet-MLX.

        Note: Parakeet doesn't use language, temperature, or prompt
        parameters, but they're accepted for API compatibility.
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
from . import BackendCapabilities, BackendDescriptor, register_backend

PARAKEET_DESCRIPTOR = BackendDescriptor(
    id="parakeet-mlx",
    display_name="Parakeet MLX",
    model_patterns=["parakeet-*", "mlx-community/parakeet-*"],
    device_types=["metal"],
    optional_dependencies=["parakeet-mlx"],
    capabilities=BackendCapabilities(
        supports_word_timestamps=True,
        supports_segments=True,
        supports_languages=["en"],
    ),
    metadata={
        "family": "parakeet",
        "precision": "bf16",
        "source": "default",
    },
)

register_backend(PARAKEET_DESCRIPTOR, _create_parakeet_backend)
