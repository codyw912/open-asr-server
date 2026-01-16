import unittest

try:
    from fastapi.testclient import TestClient
except Exception:
    TestClient = None

if TestClient is None:
    raise unittest.SkipTest("fastapi not installed")

from openai_asr_server.app import create_app
from openai_asr_server.backends.base import TranscriptionResult
from openai_asr_server.config import ServerConfig
import openai_asr_server.backends as backends


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
    def test_transcription_route_passes_prompt(self):
        fake_backend = FakeBackend()
        backends._backends.pop("test-model", None)
        backends.register_backend("test-model", lambda _: fake_backend)

        app = create_app(ServerConfig(preload_models=[]))
        client = TestClient(app)

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
