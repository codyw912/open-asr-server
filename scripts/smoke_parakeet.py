#!/usr/bin/env python

"""Smoke test for Parakeet backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_asr_server.backends.parakeet import ParakeetBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Parakeet backend")
    parser.add_argument(
        "--audio",
        default="audio_files/harvard.wav",
        help="Path to an audio file",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        help="Model ID or alias",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    backend = ParakeetBackend(args.model)
    try:
        result = backend.transcribe(audio_path)
    except ModuleNotFoundError as exc:
        if exc.name in {"parakeet_mlx", "mlx"}:
            raise SystemExit(
                "parakeet-mlx is not installed. Run with the parakeet extra:\n"
                "  uv run --extra parakeet scripts/smoke_parakeet.py"
            )
        raise

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
