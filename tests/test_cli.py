import json
import os
import sys
import types

from typer.testing import CliRunner

from open_asr_server import cli


runner = CliRunner()


def test_serve_passes_args(monkeypatch):
    calls = {}

    def run(app, host=None, port=None, reload=False, factory=False):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port
        calls["reload"] = reload
        calls["factory"] = factory

    uvicorn_module = types.ModuleType("uvicorn")
    setattr(uvicorn_module, "run", run)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.delenv("OPEN_ASR_SERVER_PRELOAD", raising=False)

    result = runner.invoke(
        cli.app,
        ["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"],
    )

    assert result.exit_code == 0
    assert calls == {
        "app": "open_asr_server.app:create_app",
        "host": "0.0.0.0",
        "port": 9000,
        "reload": True,
        "factory": True,
    }
    assert "OPEN_ASR_SERVER_PRELOAD" not in os.environ


def test_serve_sets_preload_env(monkeypatch):
    calls = {}

    def run(*args, **kwargs):
        calls["called"] = True

    uvicorn_module = types.ModuleType("uvicorn")
    setattr(uvicorn_module, "run", run)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.delenv("OPEN_ASR_SERVER_PRELOAD", raising=False)

    result = runner.invoke(
        cli.app,
        ["serve", "--preload", "model-a", "--preload", "model-b"],
    )

    assert result.exit_code == 0
    assert calls["called"] is True
    assert os.environ["OPEN_ASR_SERVER_PRELOAD"] == "model-a,model-b"


def test_backends_lists_install_hints(monkeypatch):
    statuses = [
        cli.BackendInstallStatus(
            backend_id="parakeet-mlx",
            display_name="Parakeet MLX",
            device_types=["metal"],
            model_patterns=["parakeet-*"],
            install_extra="parakeet-mlx",
            install_bundle="metal",
            install_python="3.11",
            install_command='uv tool install --python 3.11 "open-asr-server[parakeet-mlx]"',
            missing_distributions=[],
        ),
        cli.BackendInstallStatus(
            backend_id="nemo-parakeet",
            display_name="NeMo Parakeet",
            device_types=["cuda"],
            model_patterns=["nvidia/parakeet*"],
            install_extra="nemo",
            install_bundle="cuda",
            install_python=None,
            install_command='uv tool install "open-asr-server[nemo]"',
            missing_distributions=["nemo_toolkit", "torch"],
        ),
    ]
    monkeypatch.setattr(cli, "_collect_backend_statuses", lambda: statuses)

    result = runner.invoke(cli.app, ["backends"])

    assert result.exit_code == 0
    assert "Registered backends:" in result.output
    assert "parakeet-mlx (Parakeet MLX) [metal] - ready" in result.output
    assert (
        'install: uv tool install --python 3.11 "open-asr-server[parakeet-mlx]"'
        in result.output
    )
    assert "bundle: metal" in result.output
    assert "nemo-parakeet (NeMo Parakeet) [cuda] - missing deps" in result.output
    assert "missing: nemo_toolkit, torch" in result.output


def test_backends_json_output(monkeypatch):
    statuses = [
        cli.BackendInstallStatus(
            backend_id="faster-whisper",
            display_name="Faster Whisper",
            device_types=["cpu"],
            model_patterns=["openai/whisper-*"],
            install_extra="faster-whisper",
            install_bundle="cpu",
            install_python=None,
            install_command='uv tool install "open-asr-server[faster-whisper]"',
            missing_distributions=[],
        )
    ]
    monkeypatch.setattr(cli, "_collect_backend_statuses", lambda: statuses)

    result = runner.invoke(cli.app, ["backends", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["data"]) == 1
    entry = payload["data"][0]
    assert entry["backend"] == "faster-whisper"
    assert entry["available"] is True
    assert entry["install_bundle"] == "cpu"


def test_doctor_recommends_metal_on_apple_silicon(monkeypatch):
    statuses = [
        cli.BackendInstallStatus(
            backend_id="whisper-mlx",
            display_name="Whisper MLX",
            device_types=["metal"],
            model_patterns=["whisper-large-v3-turbo"],
            install_extra="whisper-mlx",
            install_bundle="metal",
            install_python="3.11",
            install_command='uv tool install --python 3.11 "open-asr-server[whisper-mlx]"',
            missing_distributions=["mlx-whisper"],
        )
    ]
    monkeypatch.setattr(cli, "_collect_backend_statuses", lambda: statuses)
    monkeypatch.setattr(cli, "_detect_nvidia_gpu", lambda: (False, "none"))
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Environment:" in result.output
    assert "Recommended quickstart:" in result.output
    assert "--python 3.11" in result.output
    assert "open-asr-server[metal]" in result.output
    assert "Backend status:" in result.output


def test_doctor_json_output(monkeypatch):
    statuses = [
        cli.BackendInstallStatus(
            backend_id="nemo-parakeet",
            display_name="NeMo Parakeet",
            device_types=["cuda"],
            model_patterns=["nvidia/parakeet*"],
            install_extra="nemo",
            install_bundle="cuda",
            install_python=None,
            install_command='uv tool install "open-asr-server[nemo]"',
            missing_distributions=[],
        )
    ]
    monkeypatch.setattr(cli, "_collect_backend_statuses", lambda: statuses)
    monkeypatch.setattr(cli, "_detect_nvidia_gpu", lambda: (True, "torch"))
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")

    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["environment"]["platform"] == "linux"
    assert payload["environment"]["nvidia_gpu"] is True
    assert payload["recommendation"]["extra"] == "nemo"
    assert (
        payload["recommendation"]["install_command"]
        == 'uv tool install "open-asr-server[nemo]"'
    )
    assert payload["backends"][0]["backend"] == "nemo-parakeet"
