# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Add backend compatibility metadata (platform, Python, GPU requirements) to install hints and `/v1/models/metadata` output.
- Improve CLI compatibility UX with richer backend status classification (`ready`, `missing_deps`, `python_incompatible`, `platform_incompatible`, `requires_gpu`).
- Extend `setup`/`doctor` flows to auto-pin known-good Python versions for CPU and CUDA stacks.
- Unify uv index configuration in `pyproject.toml` and remove `uv.toml` to eliminate ambiguous index warnings.
- Return structured runtime compatibility API errors (`backend_compatibility_error`) with compatibility context and install guidance.
- Add an optional manual self-hosted NVIDIA GPU E2E workflow (`.github/workflows/e2e-gpu.yml`) for real NeMo transcription smoke validation.

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
- Add `open-asr-server doctor` and `open-asr-server backends` commands to surface install recommendations and backend dependency status.
- Add machine-readable `--json` output for `doctor` and `backends` commands.
- Add `open-asr-server setup` command to auto-apply backend installs (including required Python version pins) without manual `--python` flags.
- Emit structured backend load errors with explicit codes (`weights_only_incompat`, `model_load_oom`, `backend_busy`) and retryable flags for gateway handling.
- Add install hints in `/v1/models/metadata` (`install_extra`, `install_bundle`, `install_python`, `install_command`) for automation.
- Harden CI with lockfile/build gates and add backend profile smoke jobs (`cpu`, `nemo`) for dependency/import validation.
- Add deterministic live-HTTP integration tests and a dedicated CI integration lane.

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
