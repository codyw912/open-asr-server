# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- TBD.

## 0.3.0

- Breaking: rename MLX extras to explicit backend names:
  - `parakeet` -> `parakeet-mlx`
  - `whisper` -> `whisper-mlx`
  - `lightning-whisper` -> `lightning-whisper-mlx`
- Add hardware bundle extras to simplify first-run setup:
  - `metal` (MLX backends)
  - `cpu` (faster-whisper + whisper.cpp)
  - `cuda` (CUDA dependency bundle for NeMo/Torch)
- Rework install docs around hardware-first quickstart paths and update smoke commands to canonical extras.

## 0.2.1

- Derive module and app version metadata from installed package metadata to prevent version drift.
- Harden NeMo model loading by narrowing `weights_only` fallback handling and preserving root-cause details when fallback fails.
- Return clearer load-time API errors by mapping retryable backend load failures to HTTP 503.

## 0.2.0

- Add NVIDIA NeMo backend scaffold, smoke script, and docs for CUDA setup.
- Add model eviction controls, admin unload/status endpoints, and CUDA graph disable toggle.

## 0.1.3

- Map faster-whisper model IDs for OpenAI/Distil Whisper patterns and guard batch_size support.
- Expand backend coverage for metadata, conflicts, and backend-specific behaviors.
- Add faster-whisper smoke test using bundled sample audio.

## 0.1.2

- Add optional in-process rate limiting and transcription timeouts with worker pools.
- Add Hugging Face token support for gated/private model downloads.
- Include sample audio clips and smoke test scripts for quick validation.
- Improve CI coverage reporting and test hermeticity guidance.

## 0.1.1

- Align MLX backends with Hugging Face cache behavior and add a model cache override.
- Improve backend coverage and add CLI smoke tests.
- Add lightweight security hardening and request guards.
