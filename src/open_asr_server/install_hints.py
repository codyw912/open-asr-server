"""Install hint helpers shared across CLI and API metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendInstallHint:
    extra: str
    bundle: str | None = None
    python: str | None = None


@dataclass(frozen=True)
class BundleInstallHint:
    extra: str
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

_BUNDLE_INSTALL_HINTS = {
    "metal": BundleInstallHint(extra="metal", python="3.11"),
    "cpu": BundleInstallHint(extra="cpu"),
    "cuda": BundleInstallHint(extra="cuda"),
}


def backend_install_hint(backend_id: str) -> BackendInstallHint | None:
    return _BACKEND_INSTALL_HINTS.get(backend_id)


def bundle_install_hint(bundle: str) -> BundleInstallHint | None:
    return _BUNDLE_INSTALL_HINTS.get(bundle)


def known_backends() -> set[str]:
    return set(_BACKEND_INSTALL_HINTS)


def known_bundles() -> set[str]:
    return set(_BUNDLE_INSTALL_HINTS)


def known_extras() -> set[str]:
    extras = {hint.extra for hint in _BACKEND_INSTALL_HINTS.values()}
    extras.update(hint.extra for hint in _BUNDLE_INSTALL_HINTS.values())
    return extras


def recommended_python_for_extra(extra: str) -> str | None:
    bundle = _BUNDLE_INSTALL_HINTS.get(extra)
    if bundle:
        return bundle.python
    for hint in _BACKEND_INSTALL_HINTS.values():
        if hint.extra == extra and hint.python is not None:
            return hint.python
    return None


def install_command(extra: str, *, python: str | None = None) -> str:
    return install_command_for_extras([extra], python=python)


def install_command_args(extra: str, *, python: str | None = None) -> list[str]:
    return install_command_args_for_extras([extra], python=python)


def install_command_for_extras(extras: list[str], *, python: str | None = None) -> str:
    extra_spec = ",".join(sorted(dict.fromkeys(extras)))
    if python:
        return f'uv tool install --python {python} "open-asr-server[{extra_spec}]"'
    return f'uv tool install "open-asr-server[{extra_spec}]"'


def install_command_args_for_extras(
    extras: list[str], *, python: str | None = None
) -> list[str]:
    args = ["uv", "tool", "install"]
    if python:
        args.extend(["--python", python])
    extra_spec = ",".join(sorted(dict.fromkeys(extras)))
    args.append(f"open-asr-server[{extra_spec}]")
    return args
