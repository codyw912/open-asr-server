# Open ASR Server

[![CI](https://github.com/codyw912/open-asr-server/actions/workflows/ci.yml/badge.svg)](https://github.com/codyw912/open-asr-server/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/open-asr-server)](https://pypi.org/project/open-asr-server/)

OpenAI-compatible ASR server with pluggable local transcription backends.

## Install

Base install includes the API server and shared models/formatters:

```bash
uv tool install "open-asr-server"
```

Add backend extras as needed:

```bash
uv tool install "open-asr-server[parakeet]"         # MLX Parakeet (Apple Silicon)
uv tool install "open-asr-server[whisper]"          # MLX Whisper
uv tool install "open-asr-server[lightning-whisper]" # MLX Lightning Whisper
uv tool install "open-asr-server[kyutai-mlx]"        # Kyutai STT (MLX)
uv tool install "open-asr-server[faster-whisper]"    # CPU (CTranslate2)
uv tool install "open-asr-server[whisper-cpp]"       # CPU (whisper.cpp)
uv tool install "open-asr-server[nemo]"              # NVIDIA NeMo (CUDA)
```

Notes:
- MLX Whisper/Lightning/Parakeet extras are currently pinned to Python 3.11.
- Kyutai MLX is currently pinned to Python 3.12.

### MLX troubleshooting

If MLX extras fail on newer Python versions, use Python 3.11 for Parakeet/Whisper/Lightning:

```bash
uv run --python 3.11 --extra whisper -- open-asr-server serve --host 127.0.0.1 --port 8000
```

For Kyutai MLX, use Python 3.12:

```bash
uv run --python 3.12 --extra kyutai-mlx -- open-asr-server serve --host 127.0.0.1 --port 8000
```

### CUDA setup (NeMo)

NeMo requires a CUDA-enabled PyTorch build. Use the PyTorch install selector to
find the right index URL for your CUDA version:

https://pytorch.org/get-started/locally/

Example (CUDA 12.1):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install "open-asr-server[nemo]"
```

Alternative (auto-detect CUDA with uv):

```bash
uv pip install torch --torch-backend=auto
uv pip install "open-asr-server[nemo]"
```

Repo-based installs can optionally use `tool.uv.sources` to route torch downloads
to a CUDA index automatically when the `nemo` extra is enabled (see
`pyproject.toml`). This only applies when working from the repo (not a PyPI
install).

Install the CUDA-enabled torch build before the `nemo` extra to avoid pulling in
a CPU-only torch dependency.

NeMo expects mono audio; the backend uses ffmpeg to downmix or convert inputs to
16kHz mono WAV when needed. Ensure ffmpeg is available in your environment.

If you see CUDA graph capture errors from NeMo decoding, set
`OPEN_ASR_NEMO_DISABLE_CUDA_GRAPHS=1` (default behavior disables CUDA graphs).

Tip: CUDA backends are often easiest to run in Docker with the NVIDIA Container
Toolkit; we do not ship a container image yet, but this keeps CUDA deps isolated.

Example Docker workflow (Linux):

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace -w /workspace \
  nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
  bash -lc "\
    apt-get update && apt-get install -y curl python3 python3-venv && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH=\"$HOME/.local/bin:$PATH\" && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install '.[nemo]' && \
    python scripts/smoke_nemo_parakeet.py samples/jfk_0_5.flac\
  "
```

Dockerfile alternative (dev-first split):

```bash
docker build -f Dockerfile.nemo.base \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t open-asr-nemo-base:torch2.5.1-cu121 .
docker build -f Dockerfile.nemo --build-arg BASE_IMAGE=open-asr-nemo-base:torch2.5.1-cu121 -t open-asr-nemo-dev:torch2.5.1-cu121 .
docker run --rm -it --gpus all -v "$(pwd)":/workspace -w /workspace open-asr-nemo-dev:torch2.5.1-cu121

docker build -f Dockerfile.nemo.base \
  --build-arg CUDA_BASE_IMAGE=pytorch/pytorch:2.5.1-cuda11.8-cudnn8-runtime \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  -t open-asr-nemo-base:torch2.5.1-cu118 .
docker build -f Dockerfile.nemo \
  --build-arg BASE_IMAGE=open-asr-nemo-base:torch2.5.1-cu118 \
  -t open-asr-nemo-dev:torch2.5.1-cu118 .
docker run --rm -it --gpus all -v "$(pwd)":/workspace -w /workspace open-asr-nemo-dev:torch2.5.1-cu118
```

Makefile helpers:

```bash
make nemo-base
make nemo-dev
make nemo-run
make nemo-base CUDA_BASE_IMAGE=pytorch/pytorch:2.5.1-cuda11.8-cudnn8-runtime BASE_TAG=torch2.5.1-cu118 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
make nemo-dev BASE_TAG=torch2.5.1-cu118 DEV_TAG=torch2.5.1-cu118
```

Docker smoke script:

```bash
scripts/smoke_nemo_parakeet_docker.sh
INFO=1 scripts/smoke_nemo_parakeet_docker.sh
HF_CACHE_DIR=/path/to/hf-cache scripts/smoke_nemo_parakeet_docker.sh
```

## Run

Install at least one backend extra before running (the default model uses
Parakeet MLX):

```bash
uv tool install "open-asr-server[parakeet]"
```

Then start the server:

```bash
uv tool run open-asr-server serve --host 127.0.0.1 --port 8000
```

Environment variables:

- `OPEN_ASR_SERVER_DEFAULT_MODEL`: default model ID for requests
- `OPEN_ASR_DEFAULT_BACKEND`: preferred backend when model patterns overlap
- `OPEN_ASR_SERVER_PRELOAD`: comma-separated models to preload at startup
- `OPEN_ASR_SERVER_API_KEY`: optional shared secret for requests
- `OPEN_ASR_SERVER_ALLOWED_MODELS`: comma-separated allowed model IDs or patterns
- `OPEN_ASR_SERVER_MAX_UPLOAD_BYTES`: max upload size in bytes (default: 26214400)
- `OPEN_ASR_SERVER_RATE_LIMIT_PER_MINUTE`: optional per-client request limit (off by default)
- `OPEN_ASR_SERVER_TRANSCRIBE_TIMEOUT_SECONDS`: optional transcription timeout (off by default)
- `OPEN_ASR_SERVER_TRANSCRIBE_WORKERS`: optional thread pool size for transcriptions
- `OPEN_ASR_SERVER_MODEL_IDLE_SECONDS`: unload models after idle timeout (off by default)
- `OPEN_ASR_SERVER_MODEL_EVICT_INTERVAL_SECONDS`: idle eviction sweep interval (default: 60)
- `OPEN_ASR_SERVER_EVICT_PRELOADED_MODELS`: allow preloaded models to be evicted (default: false)
- `OPEN_ASR_SERVER_MODEL_DIR`: override the Hugging Face cache location for this server
- `OPEN_ASR_SERVER_HF_TOKEN`: optional Hugging Face token for gated/private models

Models default to the Hugging Face cache unless a local path is provided. Use
`OPEN_ASR_SERVER_MODEL_DIR` if you want a dedicated cache without changing your
global HF environment. Use `OPEN_ASR_SERVER_HF_TOKEN` to authenticate downloads
without setting global HF environment variables.

Use `OPEN_ASR_SERVER_TRANSCRIBE_TIMEOUT_SECONDS` to bound long transcriptions.
If you set `OPEN_ASR_SERVER_TRANSCRIBE_WORKERS`, transcriptions run in a
background thread pool instead of the event loop.

Admin model management (requires API key if configured):

```bash
curl -X POST http://127.0.0.1:8000/v1/admin/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/parakeet-tdt-0.6b-v3"}'

curl -X POST http://127.0.0.1:8000/v1/admin/models/unload-all \
  -H "Content-Type: application/json" \
  -d '{"include_pinned":true}'

curl http://127.0.0.1:8000/v1/admin/models/status
```

## Sample audio

Two short clips are included in `samples/` for quick smoke tests:

- `samples/jfk_0_5.flac`
- `samples/jfk_5_10.flac`

They are derived from `tests/jfk.flac` in the OpenAI Whisper repo (MIT); the
original JFK speech is public domain.

```bash
uv run --extra parakeet scripts/smoke_parakeet.py samples/jfk_0_5.flac
uv run --python 3.11 --extra whisper scripts/smoke_whisper.py samples/jfk_0_5.flac
uv run --python 3.11 --extra lightning-whisper scripts/smoke_lightning.py samples/jfk_0_5.flac
uv run --extra whisper-cpp scripts/smoke_whisper_cpp.py samples/jfk_0_5.flac
uv run --python 3.12 --extra kyutai-mlx scripts/smoke_kyutai_mlx.py samples/jfk_0_5.flac
uv run --extra nemo scripts/smoke_nemo_parakeet.py samples/jfk_0_5.flac
```

## Backend options

Backends are selected by model ID patterns. Use `backend:model` when you need
an explicit backend.

Metal (Apple Silicon)
- Parakeet MLX: `mlx-community/parakeet-tdt-0.6b-v3` (default) or `parakeet-*`
- MLX Whisper: `whisper-large-v3-turbo` or `mlx-community/whisper-large-v3-turbo`
- Lightning Whisper MLX: `lightning-whisper-distil-large-v3`
- Kyutai STT MLX: `kyutai/stt-*-mlx`

CPU
- Faster-Whisper: `openai/whisper-*` and `distil-whisper/*`
- whisper.cpp: `tiny*`, `base*`, `small*`, `medium*`, `large*`

CUDA
- NeMo Parakeet: `nvidia/parakeet*`

## API compatibility

The server implements:

- `POST /v1/audio/transcriptions`
- `GET /v1/models`
- `GET /v1/models/metadata`

Example:

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-large-v3-turbo"
```

## Security

This server is designed for trusted networks. If you expose it publicly, enable
`OPEN_ASR_SERVER_API_KEY` and front it with a reverse proxy that provides
TLS and rate limiting. `OPEN_ASR_SERVER_RATE_LIMIT_PER_MINUTE` offers a simple
in-process limiter for single-instance use, but it is not a substitute for
production-grade rate limiting.

API key headers:

- `Authorization: Bearer <token>`
- `X-API-Key: <token>`

Use `OPEN_ASR_SERVER_ALLOWED_MODELS` to limit which model IDs can be loaded
and prevent unbounded downloads. Avoid logging request bodies or filenames if
those may contain sensitive data, and review reverse-proxy access logs for any
retention concerns.

## Release

```bash
uv version --bump patch
uv run --extra dev pytest
uv build --no-sources
uv publish --index testpypi --token "$UV_PUBLISH_TOKEN"
uv publish --token "$UV_PUBLISH_TOKEN"
```
