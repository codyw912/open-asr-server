import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import open_asr_server.backends as backends
from open_asr_server.backends import lightning_whisper, parakeet, whisper


@pytest.fixture
def reset_backend_registry():
    registered = dict(backends._registered_backends)
    instances = dict(backends._backends)
    backends._registered_backends.clear()
    backends._backends.clear()
    yield
    backends._registered_backends.clear()
    backends._registered_backends.update(registered)
    backends._backends.clear()
    backends._backends.update(instances)


def test_backend_registry_matches_patterns(reset_backend_registry):
    class DummyBackend:
        def transcribe(self, *args, **kwargs):
            raise NotImplementedError

        @property
        def supported_languages(self):
            return None

    def factory(_: str) -> DummyBackend:
        return DummyBackend()

    descriptor = backends.BackendDescriptor(
        id="test-backend",
        display_name="Test Backend",
        model_patterns=["foo-*"],
        device_types=["cpu"],
    )
    backends.register_backend(descriptor, factory)

    backend = backends.get_backend("foo-bar")

    assert isinstance(backend, DummyBackend)


def test_entry_point_load_registers_backend(reset_backend_registry):
    class DummyBackend:
        def transcribe(self, *args, **kwargs):
            raise NotImplementedError

        @property
        def supported_languages(self):
            return None

    def factory(_: str) -> DummyBackend:
        return DummyBackend()

    descriptor = backends.BackendDescriptor(
        id="entry-backend",
        display_name="Entry Backend",
        model_patterns=["entry-*"],
        device_types=["cpu"],
    )
    entry_point_value = backends.BackendEntryPoint(
        descriptor=descriptor,
        factory=factory,
    )

    class DummyEntryPoint:
        name = "dummy"

        def load(self):
            return entry_point_value

    backends._load_entry_point(DummyEntryPoint())  # type: ignore[arg-type]
    backend = backends.get_backend("entry-model")

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
    setattr(module, "from_pretrained", from_pretrained)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", module)
    monkeypatch.setenv("OPEN_ASR_SERVER_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("OPEN_ASR_SERVER_HF_TOKEN", raising=False)

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
    setattr(module, "from_pretrained", from_pretrained)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", module)

    parakeet._create_parakeet_backend("parakeet-foo")

    assert calls["model_id"] == "mlx-community/parakeet-tdt-0.6b-v3"


def test_parakeet_backend_uses_hf_token(monkeypatch, tmp_path):
    downloads = []

    def hf_hub_download(repo_id, filename, cache_dir=None, token=None):
        downloads.append((repo_id, filename, cache_dir, token))
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("config.json"):
            path.write_text("{}")
        else:
            path.write_bytes(b"")
        return str(path)

    hf_module = types.ModuleType("huggingface_hub")
    setattr(hf_module, "hf_hub_download", hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)

    class DummyArray:
        def astype(self, _dtype):
            return self

    class DummyModel:
        def __init__(self):
            self.loaded = None
            self.updated = None

        def load_weights(self, path):
            self.loaded = path

        def parameters(self):
            return {"w": DummyArray()}

        def update(self, weights):
            self.updated = weights

        def transcribe(self, _path):
            return SimpleNamespace(text="ok", sentences=[])

    def from_config(_config):
        return DummyModel()

    parakeet_utils = types.ModuleType("parakeet_mlx.utils")
    setattr(parakeet_utils, "from_config", from_config)
    monkeypatch.setitem(sys.modules, "parakeet_mlx.utils", parakeet_utils)

    def from_pretrained(*_args, **_kwargs):
        raise AssertionError("from_pretrained should not be called when token is set")

    parakeet_module = types.ModuleType("parakeet_mlx")
    setattr(parakeet_module, "from_pretrained", from_pretrained)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", parakeet_module)

    mlx_core = types.ModuleType("mlx.core")
    setattr(mlx_core, "bfloat16", "bf16")

    mlx_utils = types.ModuleType("mlx.utils")
    setattr(mlx_utils, "tree_flatten", lambda params: list(params.items()))
    setattr(mlx_utils, "tree_unflatten", lambda items: dict(items))

    mlx_pkg = types.ModuleType("mlx")
    setattr(mlx_pkg, "core", mlx_core)
    setattr(mlx_pkg, "utils", mlx_utils)

    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)

    monkeypatch.setenv("OPEN_ASR_SERVER_HF_TOKEN", "token")
    monkeypatch.setenv("OPEN_ASR_SERVER_MODEL_DIR", str(tmp_path))

    backend = parakeet.ParakeetBackend("mlx-community/parakeet-tdt-0.6b-v3")
    result = backend.transcribe(Path("audio.wav"))

    assert result.text == "ok"
    assert downloads[0][3] == "token"


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
    setattr(module, "transcribe", transcribe)
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

    def hf_hub_download(repo_id, filename, cache_dir=None, token=None):
        downloads.append((repo_id, filename, cache_dir, token))
        path = tmp_path / "hf" / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    module = types.ModuleType("huggingface_hub")
    setattr(module, "hf_hub_download", hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("OPEN_ASR_SERVER_MODEL_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OPEN_ASR_SERVER_HF_TOKEN", "token")

    transcribe_module = types.ModuleType("lightning_whisper_mlx.transcribe")

    def transcribe_audio(
        audio, *, path_or_hf_repo, batch_size, language=None, **kwargs
    ):
        calls["path_or_hf_repo"] = path_or_hf_repo
        calls["batch_size"] = batch_size
        calls["language"] = language
        calls["kwargs"] = kwargs
        return {"text": "hello", "segments": [[0, 100, "hello"]], "language": "en"}

    setattr(transcribe_module, "transcribe_audio", transcribe_audio)
    parent_module = types.ModuleType("lightning_whisper_mlx")
    setattr(parent_module, "transcribe", transcribe_module)
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
    assert downloads[0][3] == "token"


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
