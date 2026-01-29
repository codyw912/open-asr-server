# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- TBD.

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
