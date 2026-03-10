import logging
import importlib

app_module = importlib.import_module("open_asr_server.app")
from open_asr_server.config import ServerConfig


def test_startup_diagnostics_warn_on_incompatible_default_backend(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "_platform_name", lambda: "linux")
    monkeypatch.setattr(app_module, "_python_minor_version", lambda: "3.11")
    monkeypatch.setattr(app_module, "detect_nvidia_gpu", lambda: (False, "none"))

    with caplog.at_level(logging.WARNING):
        app_module._log_startup_compatibility_diagnostics(
            ServerConfig(preload_models=[])
        )

    assert "platform_incompatible" in caplog.text
    assert "open-asr-server[parakeet-mlx]" in caplog.text


def test_startup_diagnostics_info_on_compatible_default_backend(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "_platform_name", lambda: "darwin")
    monkeypatch.setattr(app_module, "_python_minor_version", lambda: "3.11")
    monkeypatch.setattr(app_module, "detect_nvidia_gpu", lambda: (False, "none"))

    with caplog.at_level(logging.INFO):
        app_module._log_startup_compatibility_diagnostics(
            ServerConfig(preload_models=[])
        )

    assert "Default backend 'parakeet-mlx' selected" in caplog.text
