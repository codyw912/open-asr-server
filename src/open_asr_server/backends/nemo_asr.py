"""NVIDIA NeMo ASR backend (CUDA)."""

from __future__ import annotations

from pathlib import Path

from .base import TranscriptionResult


def _load_nemo_model(model_id: str):
    from nemo.collections.asr.models import ASRModel  # type: ignore[import-not-found]

    if model_id.endswith(".nemo"):
        return ASRModel.restore_from(model_id)
    return ASRModel.from_pretrained(model_name=model_id)


def _audio_duration_seconds(audio_path: Path) -> float:
    try:
        import torchaudio  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return 0.0
    try:
        info = torchaudio.info(str(audio_path))
    except Exception:
        return 0.0
    if info.num_frames and info.sample_rate:
        return float(info.num_frames) / float(info.sample_rate)
    return 0.0


class NemoASRBackend:
    """CUDA-first backend powered by NVIDIA NeMo ASR."""

    def __init__(self, model_id: str = "nvidia/parakeet-tdt-0.6b-v3") -> None:
        self.model_id = model_id
        self._model = _load_nemo_model(model_id)

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        if word_timestamps:
            word_timestamps = False
        if prompt:
            prompt = None

        results = self._model.transcribe([str(audio_path)])
        text = results[0] if results else ""
        duration = _audio_duration_seconds(audio_path)
        return TranscriptionResult(
            text=text,
            language=language,
            duration=duration,
            words=None,
            segments=None,
        )

    @property
    def supported_languages(self) -> list[str] | None:
        return None


def _create_nemo_asr_backend(model_id: str) -> NemoASRBackend:
    return NemoASRBackend(model_id=model_id)


from . import BackendCapabilities, BackendDescriptor, register_backend

NEMO_ASR_DESCRIPTOR = BackendDescriptor(
    id="nemo-parakeet",
    display_name="NVIDIA NeMo Parakeet",
    model_patterns=["nvidia/parakeet*", "nvidia/multitalker-parakeet-*"],
    device_types=["cuda"],
    optional_dependencies=["nemo_toolkit[asr]"],
    capabilities=BackendCapabilities(
        supports_prompt=False,
        supports_word_timestamps=False,
        supports_segments=False,
        supports_languages=None,
    ),
    metadata={
        "family": "parakeet",
        "notes": "Requires CUDA-enabled PyTorch + NeMo ASR toolkit",
        "source": "default",
    },
)

register_backend(NEMO_ASR_DESCRIPTOR, _create_nemo_asr_backend)
