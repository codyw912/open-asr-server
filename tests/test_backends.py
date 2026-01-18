import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import open_asr_server.backends as backends
from open_asr_server.backends import lightning_whisper, parakeet, whisper


@pytest.fixture
def reset_backend_registry():
    factories = dict(backends._backend_factories)
    instances = dict(backends._backends)
    backends._backend_factories.clear()
    backends._backends.clear()
    yield
    backends._backend_factories.clear()
    backends._backend_factories.update(factories)
    backends._backends.clear()
    backends._backends.update(instances)


def test_backend_registry_matches_patterns(reset_backend_registry):
    class DummyBackend:
        pass

    def factory(_: str) -> DummyBackend:
        return DummyBackend()

    backends.register_backend("foo-*", factory)

    backend = backends.get_backend("foo-bar")

    assert isinstance(backend, DummyBackend)


def test_parakeet_backend_uses_cache_dir(monkeypatch, tmp_path):
    calls = {}

    def from_pretrained(model_id, cache_dir=None):
        calls["model_id"] = model_id
        calls["cache_dir"] = cache_dir

        def transcribe(_: str):
            token = SimpleNamespace(text="hello", start=0.0, end=0.5)
            sentence = SimpleNamespace(
                tokens=[token],
                start=0.0,
                end=0.5,
                text="hello",
                confidence=0.9,
            )
            return SimpleNamespace(text="hello", sentences=[sentence])

        return SimpleNamespace(transcribe=transcribe)

    module = types.ModuleType("parakeet_mlx")
    module.from_pretrained = from_pretrained
    monkeypatch.setitem(sys.modules, "parakeet_mlx", module)
    monkeypatch.setenv("OPEN_ASR_SERVER_MODEL_DIR", str(tmp_path))

    backend = parakeet.ParakeetBackend("mlx-community/parakeet-tdt-0.6b-v3")
    result = backend.transcribe(Path("audio.wav"), word_timestamps=True)

    assert calls["model_id"] == "mlx-community/parakeet-tdt-0.6b-v3"
    assert calls["cache_dir"] == tmp_path
    assert result.words is not None and len(result.words) == 1
    assert result.segments is not None and len(result.segments) == 1
    assert result.duration == 0.5


def test_parakeet_alias_uses_default_model(monkeypatch):
    calls = {}

    def from_pretrained(model_id, cache_dir=None):
        calls["model_id"] = model_id
        return SimpleNamespace(
            transcribe=lambda _: SimpleNamespace(text="", sentences=[])
        )

    module = types.ModuleType("parakeet_mlx")
    module.from_pretrained = from_pretrained
    monkeypatch.setitem(sys.modules, "parakeet_mlx", module)

    parakeet._create_parakeet_backend("parakeet-foo")

    assert calls["model_id"] == "mlx-community/parakeet-tdt-0.6b-v3"


def test_whisper_backend_uses_prompt_param(monkeypatch, tmp_path):
    calls = {}

    def resolve_model_path(_: str) -> Path:
        return tmp_path / "whisper"

    def transcribe(audio_path, *, path_or_hf_repo, prompt=None, **kwargs):
        calls["path_or_hf_repo"] = path_or_hf_repo
        calls["prompt"] = prompt
        calls["kwargs"] = kwargs
        return {"text": "hello", "segments": [], "language": "en"}

    module = types.ModuleType("mlx_whisper")
    module.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setattr(whisper, "resolve_model_path", resolve_model_path)
    whisper._PROMPT_PARAM = whisper._UNSET

    backend = whisper.WhisperBackend("whisper-large-v3-turbo")
    result = backend.transcribe(Path("audio.wav"), prompt="style prompt")

    assert calls["path_or_hf_repo"] == str(tmp_path / "whisper")
    assert calls["prompt"] == "style prompt"
    assert result.text == "hello"


def test_lightning_backend_distil_uses_hf_cache(monkeypatch, tmp_path):
    downloads = []
    calls = {}

    def hf_hub_download(repo_id, filename, cache_dir=None):
        downloads.append((repo_id, filename, cache_dir))
        path = tmp_path / "hf" / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("OPEN_ASR_SERVER_MODEL_DIR", str(tmp_path / "cache"))

    transcribe_module = types.ModuleType("lightning_whisper_mlx.transcribe")

    def transcribe_audio(
        audio, *, path_or_hf_repo, batch_size, language=None, **kwargs
    ):
        calls["path_or_hf_repo"] = path_or_hf_repo
        calls["batch_size"] = batch_size
        calls["language"] = language
        calls["kwargs"] = kwargs
        return {"text": "hello", "segments": [[0, 100, "hello"]], "language": "en"}

    transcribe_module.transcribe_audio = transcribe_audio
    parent_module = types.ModuleType("lightning_whisper_mlx")
    parent_module.transcribe = transcribe_module
    monkeypatch.setitem(sys.modules, "lightning_whisper_mlx", parent_module)
    monkeypatch.setitem(
        sys.modules, "lightning_whisper_mlx.transcribe", transcribe_module
    )

    backend = lightning_whisper.LightningWhisperBackend(
        "lightning-whisper-distil-large-v3"
    )
    result = backend.transcribe(Path("audio.wav"), language="en")

    assert result.text == "hello"
    assert calls["batch_size"] == 12
    assert calls["language"] == "en"
    assert Path(calls["path_or_hf_repo"]).name == "distil-large-v3"
    assert downloads[0][2] == str(tmp_path / "cache")


def test_lightning_alias_maps_repo(monkeypatch, tmp_path):
    calls = {}

    def resolve_model_path(repo_id: str) -> Path:
        calls["repo_id"] = repo_id
        return tmp_path / "model"

    monkeypatch.setattr(lightning_whisper, "resolve_model_path", resolve_model_path)

    model_path = lightning_whisper._resolve_lightning_model_path(
        "lightning-whisper-tiny"
    )

    assert calls["repo_id"] == "mlx-community/whisper-tiny"
    assert model_path == tmp_path / "model"
