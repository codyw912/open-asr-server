#!/usr/bin/env python

"""Smoke test for Kyutai MLX backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_asr_server.backends.kyutai_mlx import KyutaiMlxBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Kyutai MLX backend")
    parser.add_argument("audio", help="Path to an audio file")
    parser.add_argument(
        "--model",
        default="kyutai/stt-2.6b-en-mlx",
        help="Kyutai model ID",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    backend = KyutaiMlxBackend(model_id=args.model)
    try:
        result = backend.transcribe(audio_path)
    except ModuleNotFoundError as exc:
        if exc.name in {
            "moshi_mlx",
            "rustymimi",
            "mlx",
            "sphn",
            "sentencepiece",
            "numpy",
        }:
            raise SystemExit(
                "moshi-mlx dependencies are not installed. Run with Python 3.12 and the kyutai-mlx extra:\n"
                "  uv run --python 3.12 --extra kyutai-mlx scripts/smoke_kyutai_mlx.py"
            )
        raise

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
