#!/usr/bin/env python

"""Smoke test for MLX Whisper backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_asr_server.backends.whisper import WhisperBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test MLX Whisper backend")
    parser.add_argument(
        "--audio",
        default="audio_files/harvard.wav",
        help="Path to an audio file",
    )
    parser.add_argument(
        "--model",
        default="whisper-tiny",
        help="Model ID or alias",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    backend = WhisperBackend(args.model)
    try:
        result = backend.transcribe(audio_path)
    except ModuleNotFoundError as exc:
        if exc.name in {"mlx_whisper", "mlx"}:
            raise SystemExit(
                "mlx-whisper is not installed. Run with Python 3.11 and the whisper extra:\n"
                "  uv run --python 3.11 --extra whisper scripts/smoke_whisper.py"
            )
        raise

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
