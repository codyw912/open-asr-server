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
uv tool install "open-asr-server[parakeet]"
uv tool install "open-asr-server[whisper]"
uv tool install "open-asr-server[lightning-whisper]"
```

Note: the Whisper extras currently require Python 3.11 (tiktoken build constraints on 3.12+).

### Whisper troubleshooting

If `uv run --extra whisper` fails on Python 3.12+, use a 3.11 interpreter for now:

```bash
uv run --python 3.11 --extra whisper -- open-asr-server serve --host 127.0.0.1 --port 8000
```

## Run

```bash
uv tool run open-asr-server serve --host 127.0.0.1 --port 8000
```

Environment variables:

- `OPEN_ASR_SERVER_DEFAULT_MODEL`: default model ID for requests
- `OPEN_ASR_SERVER_PRELOAD`: comma-separated models to preload at startup
- `OPEN_ASR_SERVER_API_KEY`: optional shared secret for requests
- `OPEN_ASR_SERVER_ALLOWED_MODELS`: comma-separated allowed model IDs or patterns
- `OPEN_ASR_SERVER_MAX_UPLOAD_BYTES`: max upload size in bytes (default: 26214400)
- `OPEN_ASR_SERVER_MODEL_DIR`: override the Hugging Face cache location for this server
- `OPEN_ASR_SERVER_HF_TOKEN`: optional Hugging Face token for gated/private models

Models default to the Hugging Face cache unless a local path is provided. Use
`OPEN_ASR_SERVER_MODEL_DIR` if you want a dedicated cache without changing your
global HF environment. Use `OPEN_ASR_SERVER_HF_TOKEN` to authenticate downloads
without setting global HF environment variables.

## Backend options

All backends are MLX-based today (Apple Silicon/macOS). Non-MLX backends are
planned, but not yet supported.

Model IDs determine which backend is used:

- Parakeet MLX: `mlx-community/parakeet-tdt-0.6b-v3` (default) or `parakeet-*`
- MLX Whisper: `whisper-large-v3-turbo` or `mlx-community/whisper-large-v3-turbo`
- Lightning Whisper MLX: `lightning-whisper-distil-large-v3`

## API compatibility

The server implements:

- `POST /v1/audio/transcriptions`
- `GET /v1/models`

Example:

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-large-v3-turbo"
```

## Security

This server is designed for trusted networks. If you expose it publicly, enable
`OPEN_ASR_SERVER_API_KEY` and front it with a reverse proxy that provides
TLS and rate limiting.

API key headers:

- `Authorization: Bearer <token>`
- `X-API-Key: <token>`

Use `OPEN_ASR_SERVER_ALLOWED_MODELS` to limit which model IDs can be loaded
and prevent unbounded downloads.

## Release

```bash
uv version --bump patch
uv run --extra dev pytest
uv build --no-sources
uv publish --index testpypi --token "$UV_PUBLISH_TOKEN"
uv publish --token "$UV_PUBLISH_TOKEN"
```
