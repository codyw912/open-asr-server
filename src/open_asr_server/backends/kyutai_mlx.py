"""Kyutai STT backend using moshi-mlx (MLX)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .base import TranscriptionResult

if TYPE_CHECKING:
    import numpy as np

KYUTAI_MLX_PATTERNS = ["kyutai/stt-*-mlx"]


class KyutaiMlxBackend:
    """Kyutai STT MLX backend."""

    def __init__(self, model_id: str = "kyutai/stt-2.6b-en-mlx") -> None:
        import numpy as np
        from huggingface_hub import hf_hub_download
        import mlx.core as mx
        import mlx.nn as nn
        import rustymimi
        import sentencepiece
        import sphn

        from moshi_mlx import models, utils

        self.model_id = model_id
        self._np = np
        self._mx = mx
        self._nn = nn
        self._rustymimi = rustymimi
        self._sentencepiece = sentencepiece
        self._sphn = sphn
        self._models = models
        self._utils = utils

        mx.random.seed(299792458)

        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path, "r") as fobj:
            config = json.load(fobj)

        self._stt_config = config.get("stt_config")

        self._lm_config = models.LmConfig.from_config_dict(config)
        self._model = models.Lm(self._lm_config)
        self._model.set_dtype(mx.bfloat16)

        moshi_name = config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_hub_download(model_id, moshi_name)
        if moshi_weights.endswith(".q4.safetensors"):
            nn.quantize(self._model, bits=4, group_size=32)
        elif moshi_weights.endswith(".q8.safetensors"):
            nn.quantize(self._model, bits=8, group_size=64)

        self._model.load_weights(moshi_weights, strict=True)

        tokenizer_name = config.get("tokenizer_name", "tokenizer.model")
        tokenizer_path = hf_hub_download(model_id, tokenizer_name)
        self._text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore

        mimi_name = config.get("mimi_name", "mimi.safetensors")
        mimi_weights = hf_hub_download(model_id, mimi_name)
        generated_codebooks = self._lm_config.generated_codebooks
        other_codebooks = self._lm_config.other_codebooks
        mimi_codebooks = max(generated_codebooks, other_codebooks)
        self._other_codebooks = other_codebooks
        self._audio_tokenizer = rustymimi.Tokenizer(
            mimi_weights, num_codebooks=mimi_codebooks
        )

        if self._model.condition_provider is not None:
            self._condition_tensor = self._model.condition_provider.condition_tensor(
                "description", "very_good"
            )
        else:
            self._condition_tensor = None

        self._model.warmup(self._condition_tensor)

    def _pad_audio(self, audio):
        if not self._stt_config:
            return audio

        pad_right = float(self._stt_config.get("audio_delay_seconds", 0.0))
        pad_left = float(self._stt_config.get("audio_silence_prefix_seconds", 0.0))
        pad_left = int(pad_left * 24000)
        pad_right = int((pad_right + 1.0) * 24000)
        return self._np.pad(
            audio,
            pad_width=[(0, 0), (pad_left, pad_right)],
            mode="constant",
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file using Kyutai STT MLX."""
        if word_timestamps:
            word_timestamps = False
        if prompt:
            prompt = None

        in_pcms, _ = self._sphn.read(str(audio_path), sample_rate=24000)
        duration = float(in_pcms.shape[-1]) / 24000.0
        in_pcms = self._pad_audio(in_pcms)

        steps = int(in_pcms.shape[-1] // 1920)
        if steps <= 0:
            return TranscriptionResult(text="", language=language, duration=duration)

        text_sampler = self._utils.Sampler(top_k=25, temp=temperature)
        audio_sampler = self._utils.Sampler(top_k=250, temp=temperature)
        gen = self._models.LmGen(
            model=self._model,
            max_steps=steps,
            text_sampler=text_sampler,
            audio_sampler=audio_sampler,
            cfg_coef=1.0,
            check=False,
        )

        text_parts: list[str] = []
        for idx in range(steps):
            pcm_data = in_pcms[:, idx * 1920 : (idx + 1) * 1920]
            other_audio_tokens = self._audio_tokenizer.encode_step(pcm_data[None, 0:1])
            other_audio_tokens = self._mx.array(other_audio_tokens).transpose(0, 2, 1)[
                :, :, : self._other_codebooks
            ]
            text_token = gen.step(other_audio_tokens[0], self._condition_tensor)
            text_token = text_token[0].item()
            if text_token not in (0, 3):
                piece = self._text_tokenizer.id_to_piece(text_token)  # type: ignore
                text_parts.append(piece.replace("▁", " "))

        text = "".join(text_parts).strip()
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


def _create_kyutai_mlx_backend(model_id: str) -> KyutaiMlxBackend:
    return KyutaiMlxBackend(model_id=model_id)


# Register backends on module import
from . import BackendCapabilities, BackendDescriptor, register_backend

KYUTAI_MLX_DESCRIPTOR = BackendDescriptor(
    id="kyutai-mlx",
    display_name="Kyutai STT MLX",
    model_patterns=KYUTAI_MLX_PATTERNS,
    device_types=["metal"],
    optional_dependencies=["moshi-mlx"],
    capabilities=BackendCapabilities(
        supports_prompt=False,
        supports_word_timestamps=False,
        supports_segments=False,
        supports_languages=None,
    ),
    metadata={
        "family": "kyutai",
        "precision": "bf16",
        "notes": "Uses moshi-mlx; requires Apple Silicon + MLX",
        "source": "default",
    },
)

register_backend(KYUTAI_MLX_DESCRIPTOR, _create_kyutai_mlx_backend)
