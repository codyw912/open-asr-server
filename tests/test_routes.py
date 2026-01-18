import unittest

try:
    from fastapi.testclient import TestClient
except Exception:
    TestClient = None

if TestClient is None:
    raise unittest.SkipTest("fastapi not installed")

from fastapi.testclient import TestClient as FastAPITestClient

import open_asr_server.backends as backends
from open_asr_server.app import create_app
from open_asr_server.backends.base import TranscriptionResult
from open_asr_server.config import ServerConfig


class FakeBackend:
    def __init__(self):
        self.last_prompt = None
        self.last_language = None

    def transcribe(
        self,
        audio_path,
        language=None,
        temperature=0.0,
        word_timestamps=False,
        prompt=None,
    ):
        self.last_prompt = prompt
        self.last_language = language
        return TranscriptionResult(text="hello", language=language, duration=0.0)

    @property
    def supported_languages(self):
        return None


class RouteTests(unittest.TestCase):
    def setUp(self):
        self._factories = dict(backends._backend_factories)
        self._backends = dict(backends._backends)

    def tearDown(self):
        backends._backend_factories.clear()
        backends._backend_factories.update(self._factories)
        backends._backends.clear()
        backends._backends.update(self._backends)

    def _register_backend(self, model_id: str, backend=None):
        backend = backend or FakeBackend()
        backends._backends.pop(model_id, None)
        backends.register_backend(model_id, lambda _: backend)
        return backend

    def _client(self, config: ServerConfig):
        app = create_app(config)
        return FastAPITestClient(app)

    def test_transcription_route_passes_prompt(self):
        fake_backend = self._register_backend("test-model")
        client = self._client(ServerConfig(preload_models=[]))

        files = {"file": ("audio.wav", b"test audio", "audio/wav")}
        data = {
            "model": "test-model",
            "prompt": "style prompt",
            "language": "es",
            "response_format": "json",
        }

        response = client.post("/v1/audio/transcriptions", data=data, files=files)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "hello"})
        self.assertEqual(fake_backend.last_prompt, "style prompt")
        self.assertEqual(fake_backend.last_language, "es")

    def test_api_key_required_for_models(self):
        client = self._client(ServerConfig(preload_models=[], api_key="secret"))
        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 401)

        response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        self.assertEqual(response.status_code, 200)

    def test_model_allowlist_blocks_transcription(self):
        self._register_backend("test-model")
        client = self._client(
            ServerConfig(preload_models=[], allowed_models=["allowed-*"])
        )

        files = {"file": ("audio.wav", b"test audio", "audio/wav")}
        data = {"model": "test-model", "response_format": "json"}
        response = client.post("/v1/audio/transcriptions", data=data, files=files)

        self.assertEqual(response.status_code, 403)

    def test_model_allowlist_allows_transcription(self):
        self._register_backend("test-model")
        client = self._client(
            ServerConfig(preload_models=[], allowed_models=["test-*"])
        )

        files = {"file": ("audio.wav", b"test audio", "audio/wav")}
        data = {"model": "test-model", "response_format": "json"}
        response = client.post("/v1/audio/transcriptions", data=data, files=files)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "hello"})

    def test_max_upload_bytes_enforced(self):
        self._register_backend("test-model")
        client = self._client(ServerConfig(preload_models=[], max_upload_bytes=4))

        files = {"file": ("audio.wav", b"12345", "audio/wav")}
        data = {"model": "test-model", "response_format": "json"}
        response = client.post("/v1/audio/transcriptions", data=data, files=files)

        self.assertEqual(response.status_code, 413)

    def test_models_filtered_by_allowlist(self):
        self._register_backend("test-model")
        client = self._client(
            ServerConfig(preload_models=[], allowed_models=["test-*"])
        )

        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        model_ids = {item["id"] for item in response.json()["data"]}
        self.assertEqual(model_ids, {"test-model"})
