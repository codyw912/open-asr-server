#!/usr/bin/env python

"""Smoke test for whisper.cpp backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_asr_server.backends.whisper_cpp import WhisperCppBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test whisper.cpp backend")
    parser.add_argument("audio", help="Path to an audio file")
    parser.add_argument(
        "--model",
        default="tiny.en",
        help="whisper.cpp model name (e.g., tiny.en, base.en)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    backend = WhisperCppBackend(model_id=args.model)
    try:
        result = backend.transcribe(audio_path)
    except ModuleNotFoundError as exc:
        if exc.name in {"pywhispercpp"}:
            raise SystemExit(
                "pywhispercpp is not installed. Run with the whisper-cpp extra:\n"
                "  uv run --extra whisper-cpp scripts/smoke_whisper_cpp.py"
            )
        raise

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
