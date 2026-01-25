#!/usr/bin/env python

"""Smoke test for NVIDIA NeMo ASR backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_asr_server.backends.nemo_asr import NemoASRBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test NVIDIA NeMo ASR")
    parser.add_argument("audio", help="Path to an audio file")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="NeMo model name or .nemo file",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    try:
        backend = NemoASRBackend(model_id=args.model)
        result = backend.transcribe(audio_path)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("nemo"):
            raise SystemExit(
                "NeMo ASR dependencies are not installed. Install CUDA-enabled "
                "PyTorch first, then nemo_toolkit[asr]."
            )
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is missing. Install a CUDA-enabled torch build, then "
                "nemo_toolkit[asr]."
            )
        raise

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
