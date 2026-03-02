"""NVIDIA NeMo ASR backend (CUDA)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .base import TranscriptionResult

logger = logging.getLogger(__name__)


def _iter_exception_chain(exc: Exception):
    current = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        current = current.__context__


def _exception_summary(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_weights_only_compat_error(exc: Exception) -> bool:
    for candidate in _iter_exception_chain(exc):
        message = str(candidate).lower()
        if "weights only load failed" in message:
            return True
        if "weights_only" in message and "unpickl" in message:
            return True
    return False


def _is_retryable_load_error(exc: Exception) -> bool:
    for candidate in _iter_exception_chain(exc):
        message = str(candidate).lower()
        if any(
            token in message
            for token in (
                "out of memory",
                "cannot allocate memory",
                "resource temporarily unavailable",
                "device or resource busy",
                "cuda error",
            )
        ):
            return True
    return False


@contextmanager
def _torch_load_with_weights_only_false():
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        yield
        return

    original_load = getattr(torch, "load", None)
    if not callable(original_load):
        raise RuntimeError("torch.load is not callable")

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    setattr(torch, "load", patched_load)
    try:
        yield
    finally:
        setattr(torch, "load", original_load)


def _raise_model_load_error(
    model_id: str,
    error: Exception,
    *,
    first_error: Exception | None = None,
    fallback_attempted: bool = False,
) -> None:
    retryable = _is_retryable_load_error(error) or (
        first_error is not None and _is_retryable_load_error(first_error)
    )
    if fallback_attempted and first_error is not None:
        detail = (
            f"Failed to load NeMo model '{model_id}'. "
            "weights_only compatibility fallback also failed. "
            f"initial={_exception_summary(first_error)}; "
            f"fallback={_exception_summary(error)}"
        )
    else:
        detail = f"Failed to load NeMo model '{model_id}': {_exception_summary(error)}"
    raise BackendLoadError(
        backend_id="nemo-parakeet",
        model=model_id,
        detail=detail,
        retryable=retryable,
    ) from error


def _parse_env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _load_nemo_model(model_id: str):
    from nemo.collections.asr.models import ASRModel  # type: ignore[import-not-found]

    if model_id.endswith(".nemo"):
        try:
            return ASRModel.restore_from(model_id)
        except Exception as exc:
            _raise_model_load_error(model_id, exc)

    fallback_enabled = _parse_env_bool(
        os.getenv("OPEN_ASR_NEMO_WEIGHTS_ONLY_FALLBACK"),
        True,
    )

    try:
        return ASRModel.from_pretrained(model_name=model_id)
    except Exception as exc:
        if not (fallback_enabled and _is_weights_only_compat_error(exc)):
            _raise_model_load_error(model_id, exc)

        logger.warning(
            "NeMo load fallback: retrying %s with torch.load(weights_only=False)",
            model_id,
        )
        try:
            with _torch_load_with_weights_only_false():
                return ASRModel.from_pretrained(model_name=model_id)
        except Exception as fallback_exc:
            _raise_model_load_error(
                model_id,
                fallback_exc,
                first_error=exc,
                fallback_attempted=True,
            )


def _disable_cuda_graphs(model) -> bool:
    decoding = getattr(model, "decoding", None)
    if decoding is None:
        return False

    updated = False
    for holder in (
        decoding,
        getattr(decoding, "decoding_cfg", None),
        getattr(decoding, "cfg", None),
        getattr(model, "cfg", None),
    ):
        if holder is None:
            continue
        if hasattr(holder, "use_cuda_graphs"):
            try:
                setattr(holder, "use_cuda_graphs", False)
                updated = True
            except Exception:
                pass
        for attr in ("cuda_graphs", "enable_cuda_graphs"):
            if hasattr(holder, attr):
                try:
                    setattr(holder, attr, False)
                    updated = True
                except Exception:
                    pass
    return updated


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


def _audio_channel_count(audio_path: Path) -> int | None:
    try:
        import torchaudio  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    try:
        info = torchaudio.info(str(audio_path))
    except Exception:
        return None
    return info.num_channels


def _prepare_audio_path(
    audio_path: Path, *, force: bool = False
) -> tuple[Path, Path | None]:
    channel_count = _audio_channel_count(audio_path)
    if not force:
        if channel_count and channel_count > 1:
            logger.info("NeMo preflight: downmixing %s to mono WAV", audio_path)
        elif audio_path.suffix.lower() != ".wav":
            return audio_path, None
        elif channel_count in (None, 1):
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
        disable_graphs = _parse_env_bool(
            os.getenv("OPEN_ASR_NEMO_DISABLE_CUDA_GRAPHS"),
            True,
        )
        if disable_graphs and _disable_cuda_graphs(self._model):
            logger.info("NeMo: disabled CUDA graphs for decoding")

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

        converted_path = None
        results = []
        duration = 0.0
        try:
            audio_path, converted_path = _prepare_audio_path(audio_path)
            try:
                results = self._model.transcribe([str(audio_path)])
                duration = _audio_duration_seconds(audio_path)
            except Exception as exc:
                if converted_path is not None or audio_path.suffix.lower() == ".wav":
                    raise
                logger.warning(
                    "NeMo fallback: converting %s to WAV (%s)", audio_path, exc
                )
                audio_path, converted_path = _prepare_audio_path(audio_path, force=True)
                results = self._model.transcribe([str(audio_path)])
                duration = _audio_duration_seconds(audio_path)
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


from . import BackendCapabilities, BackendDescriptor, BackendLoadError, register_backend

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
