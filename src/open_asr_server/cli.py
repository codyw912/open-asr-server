"""CLI entry point for the ASR server."""

from dataclasses import dataclass
from importlib import metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Annotated, Optional

import typer

from .install_hints import (
    backend_install_hint,
    backend_runtime_status,
    bundle_install_hint,
    detect_nvidia_gpu,
    install_command,
    install_command_args,
    install_command_args_for_extras,
    install_command_for_extras,
    known_backends,
    known_bundles,
    known_extras,
    recommended_python_for_extra,
)

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
    install_bundle: str | None
    install_python: str | None
    install_command: str | None
    missing_distributions: list[str]
    compatibility_status: str = "ready"
    compatibility_reason: str | None = None
    supported_platforms: list[str] | None = None
    supported_python: list[str] | None = None
    requires_nvidia: bool = False
    compatibility_notes: str | None = None

    @property
    def available(self) -> bool:
        return self.compatibility_status == "ready" and not self.missing_distributions


@dataclass(frozen=True)
class QuickstartRecommendation:
    extra: str
    python: str | None = None


@dataclass(frozen=True)
class SetupPlan:
    requested_targets: list[str]
    resolved_extras: list[str]
    resolved_python: str | None

    @property
    def install_command(self) -> str:
        return install_command_for_extras(
            self.resolved_extras,
            python=self.resolved_python,
        )

    @property
    def install_command_args(self) -> list[str]:
        return install_command_args_for_extras(
            self.resolved_extras,
            python=self.resolved_python,
        )


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

    platform_name = platform.system().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    has_nvidia, _gpu_source = _detect_nvidia_gpu()

    statuses: list[BackendInstallStatus] = []
    for descriptor in sorted(list_backend_descriptors(), key=lambda item: item.id):
        hint = backend_install_hint(descriptor.id)
        compatibility_status = "ready"
        compatibility_reason = None
        supported_platforms = None
        supported_python = None
        requires_nvidia = False
        compatibility_notes = None
        if hint:
            compatibility_status, compatibility_reason = backend_runtime_status(
                descriptor.id,
                platform_name=platform_name,
                python_version=python_version,
                has_nvidia=has_nvidia,
            )
            if hint.compatibility.supported_platforms:
                supported_platforms = list(hint.compatibility.supported_platforms)
            if hint.compatibility.supported_python:
                supported_python = list(hint.compatibility.supported_python)
            requires_nvidia = hint.compatibility.requires_nvidia
            compatibility_notes = hint.compatibility.notes
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
                install_extra=hint.extra if hint else None,
                install_bundle=hint.bundle if hint else None,
                install_python=hint.python if hint else None,
                install_command=(
                    install_command(hint.extra, python=hint.python) if hint else None
                ),
                compatibility_status=compatibility_status,
                compatibility_reason=compatibility_reason,
                supported_platforms=supported_platforms,
                supported_python=supported_python,
                requires_nvidia=requires_nvidia,
                compatibility_notes=compatibility_notes,
                missing_distributions=missing,
            )
        )
    return statuses


def _detect_nvidia_gpu() -> tuple[bool, str]:
    return detect_nvidia_gpu()


def _recommend_quickstart_extra() -> QuickstartRecommendation:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin" and machine in {"arm64", "aarch64"}:
        hint = bundle_install_hint("metal")
        return QuickstartRecommendation(
            extra="metal",
            python=hint.python if hint else "3.11",
        )

    has_nvidia, _source = _detect_nvidia_gpu()
    if has_nvidia and system in {"linux", "windows"}:
        hint = backend_install_hint("nemo-parakeet")
        return QuickstartRecommendation(
            extra="nemo",
            python=hint.python if hint else None,
        )

    hint = bundle_install_hint("cpu")
    return QuickstartRecommendation(
        extra="cpu",
        python=hint.python if hint else None,
    )


def _available_setup_targets() -> list[str]:
    targets = {"auto"}
    targets.update(known_bundles())
    targets.update(known_extras())
    targets.update(known_backends())
    return sorted(targets)


def _resolve_setup_plan(targets: list[str] | None) -> SetupPlan:
    normalized_targets = [
        item.strip().lower() for item in (targets or ["auto"]) if item.strip()
    ]
    if not normalized_targets:
        normalized_targets = ["auto"]

    if "auto" in normalized_targets and len(normalized_targets) > 1:
        raise typer.BadParameter(
            "Target 'auto' cannot be combined with other targets.",
            param_hint="targets",
        )

    resolved_extras: set[str] = set()
    resolved_pythons: set[str] = set()

    for target in normalized_targets:
        if target == "auto":
            recommendation = _recommend_quickstart_extra()
            resolved_extras.add(recommendation.extra)
            if recommendation.python:
                resolved_pythons.add(recommendation.python)
            continue

        bundle_hint = bundle_install_hint(target)
        if bundle_hint:
            resolved_extras.add(bundle_hint.extra)
            if bundle_hint.python:
                resolved_pythons.add(bundle_hint.python)
            continue

        backend_hint = backend_install_hint(target)
        if backend_hint:
            resolved_extras.add(backend_hint.extra)
            if backend_hint.python:
                resolved_pythons.add(backend_hint.python)
            continue

        if target in known_extras():
            resolved_extras.add(target)
            python = recommended_python_for_extra(target)
            if python:
                resolved_pythons.add(python)
            continue

        choices = ", ".join(_available_setup_targets())
        raise typer.BadParameter(
            f"Unknown setup target '{target}'. Valid targets: {choices}",
            param_hint="targets",
        )

    if len(resolved_pythons) > 1:
        conflict = ", ".join(sorted(resolved_pythons))
        raise typer.BadParameter(
            "Conflicting recommended Python versions for selected targets: "
            + conflict
            + ". Use separate environments per backend family.",
            param_hint="targets",
        )

    resolved_python = next(iter(resolved_pythons), None)

    return SetupPlan(
        requested_targets=normalized_targets,
        resolved_extras=sorted(resolved_extras),
        resolved_python=resolved_python,
    )


def _setup_apply_command(plan: SetupPlan) -> list[str]:
    return plan.install_command_args


def _format_quickstart_install_command(recommendation: QuickstartRecommendation) -> str:
    return install_command(recommendation.extra, python=recommendation.python)


def _backend_status_payload(status: BackendInstallStatus) -> dict:
    return {
        "backend": status.backend_id,
        "display_name": status.display_name,
        "device_types": status.device_types,
        "model_patterns": status.model_patterns,
        "available": status.available,
        "status": (
            status.compatibility_status
            if status.compatibility_status != "ready"
            else ("missing_deps" if status.missing_distributions else "ready")
        ),
        "status_reason": status.compatibility_reason,
        "missing_distributions": status.missing_distributions,
        "install_extra": status.install_extra,
        "install_bundle": status.install_bundle,
        "install_python": status.install_python,
        "install_command": status.install_command,
        "supported_platforms": status.supported_platforms,
        "supported_python": status.supported_python,
        "requires_nvidia": status.requires_nvidia,
        "compatibility_notes": status.compatibility_notes,
        "setup_command": f"open-asr-server setup {status.backend_id} --apply",
    }


def _doctor_payload(
    *,
    has_nvidia: bool,
    gpu_source: str,
    recommendation: QuickstartRecommendation,
    statuses: list[BackendInstallStatus],
) -> dict:
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return {
        "environment": {
            "platform": platform.system().lower(),
            "machine": platform.machine().lower(),
            "python": python_version,
            "nvidia_gpu": has_nvidia,
            "nvidia_gpu_source": gpu_source,
        },
        "recommendation": {
            "extra": recommendation.extra,
            "python": recommendation.python,
            "install_command": _format_quickstart_install_command(recommendation),
            "setup_command": "open-asr-server setup --apply",
            "run_command": "uv tool run open-asr-server serve --host 127.0.0.1 --port 8000",
        },
        "backends": [_backend_status_payload(status) for status in statuses],
    }


def _render_backend_statuses(statuses: list[BackendInstallStatus]) -> None:
    for status in statuses:
        if status.compatibility_status != "ready":
            state_label = status.compatibility_status.replace("_", " ")
        elif status.missing_distributions:
            state_label = "missing deps"
        else:
            state_label = "ready"
        typer.echo(
            f"- {status.backend_id} ({status.display_name}) [{', '.join(status.device_types)}] - {state_label}"
        )
        typer.echo(f"  models: {', '.join(status.model_patterns)}")
        if status.compatibility_reason:
            typer.echo(f"  compatibility: {status.compatibility_reason}")
        if status.compatibility_notes:
            typer.echo(f"  notes: {status.compatibility_notes}")
        if status.install_command:
            typer.echo(f"  install: {status.install_command}")
            typer.echo(f"  setup: open-asr-server setup {status.backend_id} --apply")
        if status.install_bundle:
            typer.echo(f"  bundle: {status.install_bundle}")
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


@app.command()
def setup(
    targets: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Setup target: auto, bundle (metal/cpu/cuda), backend id "
                "(for example nemo-parakeet), or extra name"
            )
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Execute the install command automatically",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON",
        ),
    ] = False,
):
    """Generate or run a friction-free backend setup command."""
    plan = _resolve_setup_plan(targets)
    payload = {
        "requested_targets": plan.requested_targets,
        "resolved_extras": plan.resolved_extras,
        "resolved_python": plan.resolved_python,
        "install_command": plan.install_command,
    }

    if json_output and not apply:
        typer.echo(json.dumps(payload))
        return

    if not apply:
        apply_args = ""
        if plan.requested_targets != ["auto"]:
            apply_args = " " + " ".join(plan.requested_targets)
        typer.echo("Setup plan:")
        typer.echo(f"- targets: {', '.join(plan.requested_targets)}")
        typer.echo(f"- extras: {', '.join(plan.resolved_extras)}")
        if plan.resolved_python:
            typer.echo(f"- python: {plan.resolved_python}")
        typer.echo(f"- install: {plan.install_command}")
        typer.echo(f"- run: open-asr-server setup{apply_args} --apply")
        return

    if shutil.which("uv") is None:
        if json_output:
            payload["error"] = "uv not found in PATH"
            typer.echo(json.dumps(payload))
        else:
            typer.echo(
                "Error: uv not found in PATH. Install uv first: https://docs.astral.sh/uv/"
            )
        raise typer.Exit(code=1)

    result = subprocess.run(_setup_apply_command(plan), check=False)
    if json_output:
        payload["returncode"] = result.returncode
        typer.echo(json.dumps(payload))
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@app.command("backends")
def list_backends_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON",
        ),
    ] = False,
):
    """Show backend availability, model patterns, and install hints."""
    statuses = _collect_backend_statuses()
    if not statuses:
        if json_output:
            typer.echo(
                json.dumps(
                    {"error": "No backends are registered.", "data": []},
                )
            )
        else:
            typer.echo("No backends are registered.")
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps({"data": [_backend_status_payload(s) for s in statuses]}))
        return

    typer.echo("Registered backends:")
    _render_backend_statuses(statuses)


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON",
        ),
    ] = False,
):
    """Inspect local environment and print recommended install/run commands."""
    has_nvidia, gpu_source = _detect_nvidia_gpu()
    recommendation = _recommend_quickstart_extra()
    statuses = _collect_backend_statuses()
    payload = _doctor_payload(
        has_nvidia=has_nvidia,
        gpu_source=gpu_source,
        recommendation=recommendation,
        statuses=statuses,
    )

    if json_output:
        typer.echo(json.dumps(payload))
        return

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
    typer.echo(f"- install: {_format_quickstart_install_command(recommendation)}")
    typer.echo("- setup: open-asr-server setup --apply")
    typer.echo("- run: uv tool run open-asr-server serve --host 127.0.0.1 --port 8000")

    if statuses:
        typer.echo("")
        typer.echo("Backend status:")
        _render_backend_statuses(statuses)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
