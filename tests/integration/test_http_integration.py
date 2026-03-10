from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn

import open_asr_server.backends as backends
from open_asr_server.app import create_app
from open_asr_server.backends.base import TranscriptionResult
from open_asr_server.config import ServerConfig


class _IntegrationBackend:
    def __init__(self):
        self.last_prompt: str | None = None
        self.last_language: str | None = None

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        self.last_prompt = prompt
        self.last_language = language
        return TranscriptionResult(
            text="integration-ok", language=language, duration=0.0
        )


@pytest.fixture(autouse=True)
def _restore_backend_registry():
    registered = dict(backends._registered_backends)
    cache = backends._backend_cache
    cache_snapshot = dict(cache)
    yield
    backends._registered_backends.clear()
    backends._registered_backends.update(registered)
    cache.clear()
    cache.update(cache_snapshot)


def _register_backend(
    model_pattern: str,
    *,
    backend_id: str,
    factory,
) -> None:
    descriptor = backends.BackendDescriptor(
        id=backend_id,
        display_name="Integration Backend",
        model_patterns=[model_pattern],
        device_types=["cpu"],
        capabilities=backends.BackendCapabilities(
            supports_prompt=True,
            supports_word_timestamps=True,
            supports_segments=True,
        ),
    )
    backends._registered_backends.pop(backend_id, None)
    backends._backend_cache.pop((backend_id, model_pattern), None)
    backends.register_backend(descriptor, factory)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _running_server(config: ServerConfig):
    app = create_app(config)
    port = _free_port()
    uv_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(uv_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)

    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        raise RuntimeError("Timed out waiting for integration server startup")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.mark.integration
def test_transcription_endpoint_round_trip_over_http():
    backend = _IntegrationBackend()
    _register_backend(
        "itest-model",
        backend_id="itest-backend",
        factory=lambda _model_id: backend,
    )

    with _running_server(ServerConfig(preload_models=[])) as base_url:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "itest-model",
                    "response_format": "json",
                    "prompt": "style-me",
                    "language": "en",
                },
                files={"file": ("audio.wav", b"fake audio", "audio/wav")},
            )

    assert response.status_code == 200
    assert response.json() == {"text": "integration-ok"}
    assert backend.last_prompt == "style-me"
    assert backend.last_language == "en"


@pytest.mark.integration
def test_transcription_endpoint_returns_structured_load_error_over_http():
    def failing_factory(model_id: str):
        raise backends.ModelLoadOOMError(
            backend_id="itest-failing",
            model=model_id,
            detail="Out of memory while loading integration backend",
        )

    _register_backend(
        "itest-fail-model",
        backend_id="itest-failing",
        factory=failing_factory,
    )

    with _running_server(ServerConfig(preload_models=[])) as base_url:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "itest-fail-model", "response_format": "json"},
                files={"file": ("audio.wav", b"fake audio", "audio/wav")},
            )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["type"] == "backend_load_error"
    assert detail["code"] == "model_load_oom"
    assert detail["retryable"] is True


@pytest.mark.integration
def test_models_endpoint_enforces_api_key_over_http():
    _register_backend(
        "itest-auth-model",
        backend_id="itest-auth",
        factory=lambda _model_id: _IntegrationBackend(),
    )

    with _running_server(ServerConfig(preload_models=[], api_key="secret")) as base_url:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            unauthorized = client.get("/v1/models")
            authorized = client.get(
                "/v1/models",
                headers={"Authorization": "Bearer secret"},
            )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
