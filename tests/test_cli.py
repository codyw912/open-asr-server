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
    uvicorn_module.run = run
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
    uvicorn_module.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.delenv("OPEN_ASR_SERVER_PRELOAD", raising=False)

    result = runner.invoke(
        cli.app,
        ["serve", "--preload", "model-a", "--preload", "model-b"],
    )

    assert result.exit_code == 0
    assert calls["called"] is True
    assert os.environ["OPEN_ASR_SERVER_PRELOAD"] == "model-a,model-b"
