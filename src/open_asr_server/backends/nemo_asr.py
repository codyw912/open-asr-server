"""NVIDIA NeMo ASR backend (CUDA)."""

from __future__ import annotations

import subprocess
import tempfile
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


def _prepare_audio_path(audio_path: Path) -> tuple[Path, Path | None]:
    if audio_path.suffix.lower() == ".wav":
        return audio_path, None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(temp_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg is required to convert audio for the NeMo backend."
        ) from exc
    except subprocess.CalledProcessError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Failed to convert audio with ffmpeg.") from exc

    return temp_path, temp_path


def _normalize_transcript(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
    return str(value)


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

        results = []
        duration = 0.0
        try:
            results = self._model.transcribe([str(audio_path)])
            duration = _audio_duration_seconds(audio_path)
        except Exception as exc:
            if audio_path.suffix.lower() == ".wav":
                raise
            converted_path = None
            try:
                audio_path, converted_path = _prepare_audio_path(audio_path)
                results = self._model.transcribe([str(audio_path)])
                duration = _audio_duration_seconds(audio_path)
            except Exception as fallback_exc:
                raise fallback_exc from exc
            finally:
                if converted_path is not None:
                    converted_path.unlink(missing_ok=True)
        text = _normalize_transcript(results[0]) if results else ""
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
