"""CLI entry point for the ASR server."""

from dataclasses import dataclass
from importlib import metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="open-asr-server",
    help="OpenAI-compatible ASR server for local transcription.",
)


@dataclass(frozen=True)
class BackendInstallStatus:
    backend_id: str
    display_name: str
    device_types: list[str]
    model_patterns: list[str]
    install_extra: str | None
    missing_distributions: list[str]

    @property
    def available(self) -> bool:
        return not self.missing_distributions


@dataclass(frozen=True)
class QuickstartRecommendation:
    extra: str
    python: str | None = None


_BACKEND_EXTRA_HINTS = {
    "parakeet-mlx": "parakeet-mlx",
    "whisper-mlx": "whisper-mlx",
    "lightning-whisper-mlx": "lightning-whisper-mlx",
    "kyutai-mlx": "kyutai-mlx",
    "faster-whisper": "faster-whisper",
    "whisper-cpp": "whisper-cpp",
    "nemo-parakeet": "nemo",
}


def _normalize_requirement_name(requirement: str) -> str:
    token = requirement.split(";", 1)[0].strip()
    match = re.match(r"^[A-Za-z0-9_.-]+", token)
    if not match:
        return token
    return match.group(0)


def _distribution_installed(requirement: str) -> bool:
    name = _normalize_requirement_name(requirement)
    candidates = {
        name,
        name.replace("_", "-"),
        name.replace("-", "_"),
    }
    for candidate in candidates:
        try:
            metadata.version(candidate)
            return True
        except metadata.PackageNotFoundError:
            continue
    return False


def _collect_backend_statuses() -> list[BackendInstallStatus]:
    from .backends import list_backend_descriptors

    statuses: list[BackendInstallStatus] = []
    for descriptor in sorted(list_backend_descriptors(), key=lambda item: item.id):
        missing = [
            _normalize_requirement_name(requirement)
            for requirement in descriptor.optional_dependencies
            if not _distribution_installed(requirement)
        ]
        statuses.append(
            BackendInstallStatus(
                backend_id=descriptor.id,
                display_name=descriptor.display_name,
                device_types=descriptor.device_types,
                model_patterns=descriptor.model_patterns,
                install_extra=_BACKEND_EXTRA_HINTS.get(descriptor.id),
                missing_distributions=missing,
            )
        )
    return statuses


def _detect_nvidia_gpu() -> tuple[bool, str]:
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


def _recommend_quickstart_extra() -> QuickstartRecommendation:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return QuickstartRecommendation(extra="metal", python="3.11")

    has_nvidia, _source = _detect_nvidia_gpu()
    if has_nvidia and system in {"linux", "windows"}:
        return QuickstartRecommendation(extra="nemo")

    return QuickstartRecommendation(extra="cpu")


def _format_install_command(recommendation: QuickstartRecommendation) -> str:
    if recommendation.python:
        return (
            f"uv tool install --python {recommendation.python} "
            f'"open-asr-server[{recommendation.extra}]"'
        )
    return f'uv tool install "open-asr-server[{recommendation.extra}]"'


def _render_backend_statuses(statuses: list[BackendInstallStatus]) -> None:
    for status in statuses:
        state_label = "ready" if status.available else "missing deps"
        typer.echo(
            f"- {status.backend_id} ({status.display_name}) [{', '.join(status.device_types)}] - {state_label}"
        )
        typer.echo(f"  models: {', '.join(status.model_patterns)}")
        if status.install_extra:
            typer.echo(
                f'  install: uv tool install "open-asr-server[{status.install_extra}]"'
            )
        if status.missing_distributions:
            typer.echo(f"  missing: {', '.join(status.missing_distributions)}")


@app.callback()
def cli():
    """OpenAI-compatible ASR server for local transcription."""


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", "-h", help="Host to bind to")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 8000,
    preload: Annotated[
        Optional[list[str]],
        typer.Option("--preload", "-m", help="Models to preload at startup"),
    ] = None,
    reload: Annotated[
        bool, typer.Option("--reload", help="Enable auto-reload for development")
    ] = False,
):
    """Start the transcription server."""
    import uvicorn

    if preload is not None:
        os.environ["OPEN_ASR_SERVER_PRELOAD"] = ",".join(preload)

    uvicorn.run(
        "open_asr_server.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@app.command("backends")
def list_backends_command():
    """Show backend availability, model patterns, and install hints."""
    statuses = _collect_backend_statuses()
    if not statuses:
        typer.echo("No backends are registered.")
        raise typer.Exit(code=1)

    typer.echo("Registered backends:")
    _render_backend_statuses(statuses)


@app.command()
def doctor():
    """Inspect local environment and print recommended install/run commands."""
    has_nvidia, gpu_source = _detect_nvidia_gpu()
    recommendation = _recommend_quickstart_extra()

    typer.echo("Environment:")
    typer.echo(f"- platform: {platform.system().lower()} {platform.machine().lower()}")
    typer.echo(
        f"- python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    typer.echo(
        f"- nvidia_gpu: {'yes' if has_nvidia else 'no'}"
        + ("" if gpu_source == "none" else f" ({gpu_source})")
    )

    typer.echo("")
    typer.echo("Recommended quickstart:")
    typer.echo(f"- install: {_format_install_command(recommendation)}")
    typer.echo("- run: uv tool run open-asr-server serve --host 127.0.0.1 --port 8000")

    statuses = _collect_backend_statuses()
    if statuses:
        typer.echo("")
        typer.echo("Backend status:")
        _render_backend_statuses(statuses)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
