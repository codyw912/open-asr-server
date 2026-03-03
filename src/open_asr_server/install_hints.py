"""Install hint helpers shared across CLI and API metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendInstallHint:
    extra: str
    bundle: str | None = None
    python: str | None = None


_BACKEND_INSTALL_HINTS = {
    "parakeet-mlx": BackendInstallHint(
        extra="parakeet-mlx",
        bundle="metal",
        python="3.11",
    ),
    "whisper-mlx": BackendInstallHint(
        extra="whisper-mlx",
        bundle="metal",
        python="3.11",
    ),
    "lightning-whisper-mlx": BackendInstallHint(
        extra="lightning-whisper-mlx",
        bundle="metal",
        python="3.11",
    ),
    "kyutai-mlx": BackendInstallHint(
        extra="kyutai-mlx",
        bundle="metal",
        python="3.12",
    ),
    "faster-whisper": BackendInstallHint(extra="faster-whisper", bundle="cpu"),
    "whisper-cpp": BackendInstallHint(extra="whisper-cpp", bundle="cpu"),
    "nemo-parakeet": BackendInstallHint(extra="nemo", bundle="cuda"),
}


def backend_install_hint(backend_id: str) -> BackendInstallHint | None:
    return _BACKEND_INSTALL_HINTS.get(backend_id)


def install_command(extra: str, *, python: str | None = None) -> str:
    if python:
        return f'uv tool install --python {python} "open-asr-server[{extra}]"'
    return f'uv tool install "open-asr-server[{extra}]"'
