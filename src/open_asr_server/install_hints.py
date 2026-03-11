"""Install hint helpers shared across CLI and API metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import shutil
import subprocess
from typing import Literal

RuntimeStatus = Literal[
    "ready",
    "platform_incompatible",
    "python_incompatible",
    "requires_gpu",
]


@dataclass(frozen=True)
class RuntimeCompatibility:
    supported_platforms: tuple[str, ...] | None = None
    supported_python: tuple[str, ...] | None = None
    requires_nvidia: bool = False
    notes: str | None = None


@dataclass(frozen=True)
class BackendInstallHint:
    extra: str
    bundle: str | None = None
    python: str | None = None
    compatibility: RuntimeCompatibility = field(default_factory=RuntimeCompatibility)


@dataclass(frozen=True)
class BundleInstallHint:
    extra: str
    python: str | None = None
    compatibility: RuntimeCompatibility = field(default_factory=RuntimeCompatibility)


_ALL_PLATFORMS = ("darwin", "linux", "windows")

_MLX_311 = RuntimeCompatibility(
    supported_platforms=("darwin",),
    supported_python=("3.11",),
    notes="Apple Silicon (Metal/MLX)",
)

_MLX_312 = RuntimeCompatibility(
    supported_platforms=("darwin",),
    supported_python=("3.12",),
    notes="Apple Silicon (Metal/MLX)",
)

_CPU_311_PLUS = RuntimeCompatibility(
    supported_platforms=_ALL_PLATFORMS,
    supported_python=("3.11", "3.12", "3.13"),
)

_CUDA_311 = RuntimeCompatibility(
    supported_platforms=("linux", "windows"),
    supported_python=("3.11",),
    requires_nvidia=True,
    notes="NVIDIA CUDA host required",
)


_BACKEND_INSTALL_HINTS = {
    "parakeet-mlx": BackendInstallHint(
        extra="parakeet-mlx",
        bundle="metal",
        python="3.11",
        compatibility=_MLX_311,
    ),
    "whisper-mlx": BackendInstallHint(
        extra="whisper-mlx",
        bundle="metal",
        python="3.11",
        compatibility=_MLX_311,
    ),
    "lightning-whisper-mlx": BackendInstallHint(
        extra="lightning-whisper-mlx",
        bundle="metal",
        python="3.11",
        compatibility=_MLX_311,
    ),
    "kyutai-mlx": BackendInstallHint(
        extra="kyutai-mlx",
        bundle="metal",
        python="3.12",
        compatibility=_MLX_312,
    ),
    "faster-whisper": BackendInstallHint(
        extra="faster-whisper",
        bundle="cpu",
        python="3.11",
        compatibility=_CPU_311_PLUS,
    ),
    "whisper-cpp": BackendInstallHint(
        extra="whisper-cpp",
        bundle="cpu",
        python="3.11",
        compatibility=_CPU_311_PLUS,
    ),
    "nemo-parakeet": BackendInstallHint(
        extra="nemo",
        bundle="cuda",
        python="3.11",
        compatibility=_CUDA_311,
    ),
}

_BUNDLE_INSTALL_HINTS = {
    "metal": BundleInstallHint(
        extra="metal",
        python="3.11",
        compatibility=RuntimeCompatibility(
            supported_platforms=("darwin",),
            supported_python=("3.11", "3.12"),
            notes="Use 3.11 for Parakeet/Whisper/Lightning; 3.12 for Kyutai",
        ),
    ),
    "cpu": BundleInstallHint(
        extra="cpu",
        python="3.11",
        compatibility=_CPU_311_PLUS,
    ),
    "cuda": BundleInstallHint(
        extra="cuda",
        python="3.11",
        compatibility=_CUDA_311,
    ),
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


def runtime_status(
    compatibility: RuntimeCompatibility,
    *,
    platform_name: str,
    python_version: str,
    has_nvidia: bool,
) -> tuple[RuntimeStatus, str | None]:
    normalized_platform = platform_name.strip().lower()

    if compatibility.supported_platforms:
        supported_platforms = {
            value.lower() for value in compatibility.supported_platforms
        }
        if normalized_platform not in supported_platforms:
            options = ", ".join(sorted(supported_platforms))
            return "platform_incompatible", f"supported platforms: {options}"

    if (
        compatibility.supported_python
        and python_version not in compatibility.supported_python
    ):
        options = ", ".join(compatibility.supported_python)
        return "python_incompatible", f"supported python: {options}"

    if compatibility.requires_nvidia and not has_nvidia:
        return "requires_gpu", "requires NVIDIA GPU"

    return "ready", None


def backend_runtime_status(
    backend_id: str,
    *,
    platform_name: str,
    python_version: str,
    has_nvidia: bool,
) -> tuple[RuntimeStatus, str | None]:
    hint = backend_install_hint(backend_id)
    if not hint:
        return "ready", None
    return runtime_status(
        hint.compatibility,
        platform_name=platform_name,
        python_version=python_version,
        has_nvidia=has_nvidia,
    )


@lru_cache(maxsize=1)
def detect_nvidia_gpu() -> tuple[bool, str]:
    try:
        import torch  # type: ignore[import-not-found]

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return True, "torch"
    except Exception:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, "nvidia-smi"
        except Exception:
            pass

    return False, "none"


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
