# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "datasets",
#   "jiwer",
#   "soundfile",
#   "parakeet-mlx",
#   "numba>=0.59",
#   "llvmlite>=0.42",
# ]
# ///
"""Quick Parakeet-MLX comparison on LibriSpeech test-clean.

Note: run with Python 3.11 to avoid numba/llvmlite resolver issues.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from jiwer import wer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_asr_server.backends.parakeet import ParakeetBackend


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick Parakeet-MLX WER/RTFx check")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--split", default="test", help="Dataset split name")
    parser.add_argument("--config", default="clean", help="LibriSpeech config")
    parser.add_argument(
        "--model-id",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        help="Parakeet model id",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before select")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print per-sample progress",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = f"{args.split}[:{args.samples}]"

    print(f"Loading LibriSpeech {args.config} ({split})...")
    dataset = load_dataset("librispeech_asr", args.config, split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.samples))

    print(f"Loading Parakeet-MLX model: {args.model_id}")
    load_start = time.perf_counter()
    backend = ParakeetBackend(model_id=args.model_id)
    load_time = time.perf_counter() - load_start

    refs: list[str] = []
    preds: list[str] = []
    durations: list[float] = []
    times: list[float] = []

    print("Starting transcription...")
    for idx, sample in enumerate(dataset, start=1):
        audio_bytes = sample["audio"]["bytes"]
        if audio_bytes is None:
            raise RuntimeError("Missing audio bytes in dataset sample")

        info = sf.info(io.BytesIO(audio_bytes))
        duration = info.frames / info.samplerate

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = Path(tmp.name)

            start = time.perf_counter()
            result = backend.transcribe(tmp_path)
            elapsed = time.perf_counter() - start
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        refs.append(sample["text"])
        preds.append(result.text)
        durations.append(duration)
        times.append(elapsed)

        if args.progress:
            print(f"{idx}/{args.samples}: {elapsed:.2f}s (audio {duration:.2f}s)")

    normalized_refs = [normalize(text) for text in refs]
    normalized_preds = [normalize(text) for text in preds]

    normalized_wer = wer(normalized_refs, normalized_preds)
    raw_wer = wer(refs, preds)
    total_audio = sum(durations)
    total_time = sum(times)
    rtfx = total_audio / total_time if total_time else 0.0

    print("\nSummary")
    print("--------")
    print(f"Samples: {args.samples}")
    print(f"Model load time: {load_time:.2f}s")
    print(f"Total audio seconds: {total_audio:.2f}s")
    print(f"Total transcribe time: {total_time:.2f}s")
    print(f"RTFx (audio/compute): {rtfx:.2f}")
    print(f"WER (normalized): {normalized_wer:.4f}")
    print(f"WER (raw): {raw_wer:.4f}")


if __name__ == "__main__":
    main()
