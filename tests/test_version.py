import importlib
import importlib.metadata

import open_asr_server
from open_asr_server.app import create_app
from open_asr_server.config import ServerConfig


def test_module_version_uses_distribution_metadata(monkeypatch):
    module = open_asr_server

    with monkeypatch.context() as context:
        context.setattr(importlib.metadata, "version", lambda _: "9.9.9")
        reloaded = importlib.reload(module)
        assert reloaded.__version__ == "9.9.9"

    importlib.reload(module)


def test_module_version_falls_back_when_metadata_missing(monkeypatch):
    module = open_asr_server

    def raise_not_found(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    with monkeypatch.context() as context:
        context.setattr(importlib.metadata, "version", raise_not_found)
        reloaded = importlib.reload(module)
        assert reloaded.__version__ == "0.0.0"

    importlib.reload(module)


def test_fastapi_uses_module_version():
    app = create_app(ServerConfig(preload_models=[]))
    assert app.version == open_asr_server.__version__
