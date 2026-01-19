#!/usr/bin/env python

"""Smoke test for Lightning Whisper backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_asr_server.backends.lightning_whisper import LightningWhisperBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Lightning Whisper backend")
    parser.add_argument(
        "audio",
        help="Path to an audio file",
    )
    parser.add_argument(
        "--model",
        default="lightning-whisper-tiny",
        help="Model ID or alias",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for decoding",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    backend = LightningWhisperBackend(args.model, batch_size=args.batch_size)
    try:
        result = backend.transcribe(audio_path)
    except ModuleNotFoundError as exc:
        if exc.name in {"lightning_whisper_mlx", "mlx"}:
            raise SystemExit(
                "lightning-whisper-mlx is not installed. Run with Python 3.11 and the lightning-whisper extra:\n"
                "  uv run --python 3.11 --extra lightning-whisper scripts/smoke_lightning.py"
            )
        raise

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
