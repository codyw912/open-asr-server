import time
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
        self._registered = dict(backends._registered_backends)
        self._cache = (
            backends._backend_cache
            if hasattr(backends, "_backend_cache")
            else backends._backends
        )
        self._backends = dict(self._cache)

    def tearDown(self):
        backends._registered_backends.clear()
        backends._registered_backends.update(self._registered)
        self._cache.clear()
        self._cache.update(self._backends)

    def _register_backend(self, model_id: str, backend=None):
        backend = backend or FakeBackend()
        backend_id = f"test-{model_id}".replace("/", "-").replace(":", "-")
        descriptor = backends.BackendDescriptor(
            id=backend_id,
            display_name="Test Backend",
            model_patterns=[model_id],
            device_types=["cpu"],
            capabilities=backends.BackendCapabilities(
                supports_prompt=True,
                supports_word_timestamps=True,
                supports_segments=True,
            ),
            metadata={"source": "default"},
        )
        backends._registered_backends.pop(backend_id, None)
        self._cache.pop((backend_id, model_id), None)
        backends.register_backend(descriptor, lambda _: backend)
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

    def test_rate_limit_enforced(self):
        client = self._client(ServerConfig(preload_models=[], rate_limit_per_minute=1))

        first = client.get("/v1/models")
        second = client.get("/v1/models")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    def test_transcription_timeout(self):
        class SlowBackend(FakeBackend):
            def transcribe(
                self,
                audio_path,
                language=None,
                temperature=0.0,
                word_timestamps=False,
                prompt=None,
            ):
                time.sleep(0.05)
                return TranscriptionResult(text="slow", language=language, duration=0.0)

        self._register_backend("test-model", SlowBackend())
        client = self._client(
            ServerConfig(preload_models=[], transcribe_timeout_seconds=0.01)
        )

        files = {"file": ("audio.wav", b"test audio", "audio/wav")}
        data = {"model": "test-model", "response_format": "json"}
        response = client.post("/v1/audio/transcriptions", data=data, files=files)

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json(), {"detail": "Transcription timed out"})

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

    def test_models_metadata_includes_patterns_and_loaded_models(self):
        descriptor = backends.BackendDescriptor(
            id="test-meta",
            display_name="Test Metadata",
            model_patterns=["test-model"],
            device_types=["cpu"],
            capabilities=backends.BackendCapabilities(
                supports_prompt=True,
                supports_word_timestamps=True,
                supports_segments=True,
            ),
            metadata={
                "family": "whisper",
                "precision": "int8",
                "min_ram_mb": 512.0,
                "notes": "metadata notes",
                "source": "default",
            },
        )
        backends.register_backend(descriptor, lambda _: FakeBackend())
        backends.get_backend("test-model")
        client = self._client(ServerConfig(preload_models=[]))

        response = client.get("/v1/models/metadata")

        self.assertEqual(response.status_code, 200)
        data = {entry["id"]: entry for entry in response.json()["data"]}
        self.assertIn("test-model", data)
        self.assertIn("test-meta:test-model", data)
        pattern_entry = data["test-model"]
        loaded_entry = data["test-meta:test-model"]
        self.assertEqual(pattern_entry["backend"], "test-meta")
        self.assertEqual(pattern_entry["precision"], "int8")
        self.assertEqual(pattern_entry["min_ram_mb"], 512.0)
        self.assertEqual(pattern_entry["notes"], "metadata notes")
        self.assertTrue(pattern_entry["capabilities"]["supports_prompt"])
        self.assertEqual(loaded_entry["device_types"], ["cpu"])

    def test_models_metadata_allowlist_accepts_prefixed_ids(self):
        descriptor = backends.BackendDescriptor(
            id="test-meta-allow",
            display_name="Test Metadata Allowlist",
            model_patterns=["allowed-model"],
            device_types=["cpu"],
        )
        backends.register_backend(descriptor, lambda _: FakeBackend())
        backends.get_backend("allowed-model")
        client = self._client(
            ServerConfig(preload_models=[], allowed_models=["allowed-model"])
        )

        response = client.get("/v1/models/metadata")

        self.assertEqual(response.status_code, 200)
        model_ids = {entry["id"] for entry in response.json()["data"]}
        self.assertEqual(model_ids, {"allowed-model", "test-meta-allow:allowed-model"})

    def test_transcription_conflict_returns_candidates(self):
        backends.register_backend(
            backends.BackendDescriptor(
                id="conflict-a",
                display_name="Conflict A",
                model_patterns=["conflict-model"],
                device_types=["cpu"],
            ),
            lambda _: FakeBackend(),
        )
        backends.register_backend(
            backends.BackendDescriptor(
                id="conflict-b",
                display_name="Conflict B",
                model_patterns=["conflict-model"],
                device_types=["cpu"],
            ),
            lambda _: FakeBackend(),
        )
        client = self._client(
            ServerConfig(preload_models=[], default_backend="missing")
        )

        files = {"file": ("audio.wav", b"test audio", "audio/wav")}
        data = {"model": "conflict-model", "response_format": "json"}
        response = client.post("/v1/audio/transcriptions", data=data, files=files)

        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertIn("conflict-a", detail)
        self.assertIn("conflict-b", detail)
        self.assertIn("OPEN_ASR_DEFAULT_BACKEND", detail)

    def test_admin_unload_model(self):
        self._register_backend("test-model")
        backends.get_backend("test-model")
        client = self._client(ServerConfig(preload_models=[]))

        response = client.post(
            "/v1/admin/models/unload",
            json={"model": "test-model"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["unloaded"], ["test-test-model:test-model"])
        self.assertEqual(payload["skipped"], [])
        self.assertEqual(payload["loaded"], [])

    def test_admin_unload_all_models_respects_pinned(self):
        self._register_backend("test-model")
        self._register_backend("test-model-2")
        backends.preload_backend("test-model")
        backends.get_backend("test-model-2")
        client = self._client(ServerConfig(preload_models=[]))

        response = client.post(
            "/v1/admin/models/unload-all",
            json={"include_pinned": False},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["unloaded"], ["test-test-model-2:test-model-2"])
        self.assertEqual(payload["skipped"], ["test-test-model:test-model"])
        self.assertEqual(payload["loaded"], ["test-test-model:test-model"])

        response = client.post(
            "/v1/admin/models/unload-all",
            json={"include_pinned": True},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("test-test-model:test-model", payload["unloaded"])
        self.assertEqual(payload["skipped"], [])
        self.assertEqual(payload["loaded"], [])
