# Blessed Environments

This document defines recommended runtime stacks for `open-asr-server`.

These recommendations mirror the compatibility metadata used by:
- `open-asr-server doctor`
- `open-asr-server backends`
- `GET /v1/models/metadata`

## Quick picks

| Goal | Command |
| --- | --- |
| Auto choose best stack for this host | `open-asr-server setup --apply` |
| Apple Silicon MLX stack | `open-asr-server setup metal --apply` |
| CPU stack | `open-asr-server setup cpu --apply` |
| NVIDIA CUDA stack | `open-asr-server setup cuda --apply` |

## Backend matrix

| Backend | Extra | Bundle | Supported OS | Supported Python | NVIDIA GPU Required | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `parakeet-mlx` | `parakeet-mlx` | `metal` | `darwin` | `3.11` | No | Apple Silicon (Metal/MLX) |
| `whisper-mlx` | `whisper-mlx` | `metal` | `darwin` | `3.11` | No | Apple Silicon (Metal/MLX) |
| `lightning-whisper-mlx` | `lightning-whisper-mlx` | `metal` | `darwin` | `3.11` | No | Apple Silicon (Metal/MLX) |
| `kyutai-mlx` | `kyutai-mlx` | `metal` | `darwin` | `3.12` | No | Apple Silicon (Metal/MLX) |
| `faster-whisper` | `faster-whisper` | `cpu` | `darwin`, `linux`, `windows` | `3.11`, `3.12`, `3.13` | No | CPU baseline |
| `whisper-cpp` | `whisper-cpp` | `cpu` | `darwin`, `linux`, `windows` | `3.11`, `3.12`, `3.13` | No | CPU baseline |
| `nemo-parakeet` | `nemo` | `cuda` | `linux`, `windows` | `3.11` | Yes | NVIDIA CUDA host required |

## Bundle matrix

| Bundle | Recommended Python | Supported OS | Notes |
| --- | --- | --- | --- |
| `metal` | `3.11` | `darwin` | Use `3.11` for Parakeet/Whisper/Lightning, `3.12` for Kyutai |
| `cpu` | `3.11` | `darwin`, `linux`, `windows` | Most stable cross-platform baseline |
| `cuda` | `3.11` | `linux`, `windows` | Requires NVIDIA GPU and CUDA-capable torch stack |

## Multi-backend setups

You can install multiple compatible stacks at once:

```bash
open-asr-server setup metal cpu --apply
```

If selected targets require conflicting Python versions, setup exits with a clear
error and asks you to use separate environments.
