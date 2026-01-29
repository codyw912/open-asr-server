import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import open_asr_server.backends as backends
from open_asr_server.backends import (
    faster_whisper,
    kyutai_mlx,
    lightning_whisper,
    nemo_asr,
    parakeet,
    whisper,
    whisper_cpp,
)


@pytest.fixture
def reset_backend_registry():
    registered = dict(backends._registered_backends)
    cache = (
        backends._backend_cache
        if hasattr(backends, "_backend_cache")
        else backends._backends
    )
    instances = dict(cache)
    backends._registered_backends.clear()
    cache.clear()
    yield
    backends._registered_backends.clear()
    backends._registered_backends.update(registered)
    cache.clear()
    cache.update(instances)


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


def test_backend_resolves_prefixed_model(reset_backend_registry):
    class DummyBackend:
        def __init__(self, backend_id: str):
            self.backend_id = backend_id

        def transcribe(self, *args, **kwargs):
            raise NotImplementedError

        @property
        def supported_languages(self):
            return None

    calls = {}

    def factory(model_id: str) -> DummyBackend:
        calls["model_id"] = model_id
        return DummyBackend("prefixed")

    descriptor = backends.BackendDescriptor(
        id="prefixed",
        display_name="Prefixed Backend",
        model_patterns=["demo-*"],
        device_types=["cpu"],
    )
    backends.register_backend(descriptor, factory)

    backend = backends.get_backend("prefixed:demo-model")

    assert isinstance(backend, DummyBackend)
    assert backend.backend_id == "prefixed"
    assert calls["model_id"] == "demo-model"


def test_backend_resolves_default_from_env(reset_backend_registry, monkeypatch):
    class DummyBackend:
        def __init__(self, backend_id: str):
            self.backend_id = backend_id

        def transcribe(self, *args, **kwargs):
            raise NotImplementedError

        @property
        def supported_languages(self):
            return None

    calls = []

    def factory(backend_id: str):
        def _factory(model_id: str) -> DummyBackend:
            calls.append((backend_id, model_id))
            return DummyBackend(backend_id)

        return _factory

    backends.register_backend(
        backends.BackendDescriptor(
            id="alpha",
            display_name="Alpha Backend",
            model_patterns=["shared-*"],
            device_types=["cpu"],
        ),
        factory("alpha"),
    )
    backends.register_backend(
        backends.BackendDescriptor(
            id="beta",
            display_name="Beta Backend",
            model_patterns=["shared-*"],
            device_types=["cpu"],
        ),
        factory("beta"),
    )
    monkeypatch.setenv("OPEN_ASR_DEFAULT_BACKEND", "beta")

    backend = backends.get_backend("shared-model")

    assert isinstance(backend, DummyBackend)
    assert backend.backend_id == "beta"
    assert calls == [("beta", "shared-model")]


def test_faster_whisper_transcribe_passes_parameters(monkeypatch):
    calls = {}

    class DummyWhisperModel:
        def __init__(self, model_id, device, compute_type):
            calls["init"] = (model_id, device, compute_type)

        def transcribe(self, audio_path, **kwargs):
            calls["audio_path"] = audio_path
            calls["kwargs"] = kwargs
            segment = SimpleNamespace(
                text=" hello ",
                start=0.0,
                end=1.0,
                words=[SimpleNamespace(word="hello", start=0.0, end=1.0)],
            )
            info = SimpleNamespace(language="en")
            return [segment], info

    module = types.ModuleType("faster_whisper")
    setattr(module, "WhisperModel", DummyWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    backend = faster_whisper.FasterWhisperBackend(
        model_id="openai/whisper-tiny",
        compute_type="int8",
        device="cpu",
        beam_size=7,
        batch_size=2,
    )
    result = backend.transcribe(
        Path("audio.wav"),
        language="es",
        temperature=0.4,
        word_timestamps=True,
        prompt="style prompt",
    )

    assert calls["init"] == ("tiny", "cpu", "int8")
    assert calls["audio_path"] == "audio.wav"
    assert calls["kwargs"] == {
        "language": "es",
        "temperature": 0.4,
        "beam_size": 7,
        "word_timestamps": True,
        "initial_prompt": "style prompt",
        "batch_size": 2,
    }
    assert result.text == "hello"
    assert result.language == "en"
    assert result.duration == 1.0
    assert result.words is not None and result.words[0].word == "hello"
    assert result.segments is not None and result.segments[0].text == "hello"


def test_faster_whisper_skips_unsupported_batch_size(monkeypatch):
    calls = {}

    class DummyWhisperModel:
        def __init__(self, model_id, device, compute_type):
            calls["init"] = (model_id, device, compute_type)

        def transcribe(
            self,
            audio_path,
            *,
            language=None,
            temperature=0.0,
            beam_size=5,
            word_timestamps=False,
            initial_prompt=None,
        ):
            calls["audio_path"] = audio_path
            calls["kwargs"] = {
                "language": language,
                "temperature": temperature,
                "beam_size": beam_size,
                "word_timestamps": word_timestamps,
                "initial_prompt": initial_prompt,
            }
            return [], SimpleNamespace(language=None)

    module = types.ModuleType("faster_whisper")
    setattr(module, "WhisperModel", DummyWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    backend = faster_whisper.FasterWhisperBackend(
        model_id="openai/whisper-tiny",
        batch_size=4,
    )
    result = backend.transcribe(
        Path("audio.wav"),
        language="fr",
        temperature=0.1,
        word_timestamps=False,
        prompt=None,
    )

    assert calls["init"] == ("tiny", "cpu", "int8")
    assert calls["audio_path"] == "audio.wav"
    assert calls["kwargs"] == {
        "language": "fr",
        "temperature": 0.1,
        "beam_size": 5,
        "word_timestamps": False,
        "initial_prompt": None,
    }
    assert result.text == ""
    assert result.duration == 0.0


def test_faster_whisper_smoke():
    pytest.importorskip("faster_whisper")
    audio_path = Path(__file__).resolve().parents[1] / "samples" / "jfk_0_5.flac"
    assert audio_path.exists()

    backend = faster_whisper.FasterWhisperBackend(model_id="openai/whisper-tiny")
    result = backend.transcribe(audio_path)

    assert result.text.strip()
    assert result.duration > 0


def test_nemo_backend_uses_from_pretrained(monkeypatch):
    calls = {}

    class DummyModel:
        def transcribe(self, audio_paths):
            calls["audio_paths"] = audio_paths
            return ["hello"]

    class DummyASRModel:
        @staticmethod
        def from_pretrained(model_name):
            calls["model_name"] = model_name
            return DummyModel()

        @staticmethod
        def restore_from(path):
            calls["restore_from"] = path
            return DummyModel()

    module = types.ModuleType("nemo.collections.asr.models")
    setattr(module, "ASRModel", DummyASRModel)
    monkeypatch.setitem(sys.modules, "nemo", types.ModuleType("nemo"))
    monkeypatch.setitem(
        sys.modules, "nemo.collections", types.ModuleType("nemo.collections")
    )
    monkeypatch.setitem(
        sys.modules, "nemo.collections.asr", types.ModuleType("nemo.collections.asr")
    )
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", module)
    monkeypatch.setattr(nemo_asr, "_audio_duration_seconds", lambda _path: 1.5)

    backend = nemo_asr.NemoASRBackend("nvidia/parakeet-test")
    result = backend.transcribe(Path("audio.wav"), language="en")

    assert calls["model_name"] == "nvidia/parakeet-test"
    assert calls["audio_paths"] == ["audio.wav"]
    assert result.text == "hello"
    assert result.language == "en"
    assert result.duration == 1.5


def test_nemo_backend_restores_nemo_file(monkeypatch):
    calls = {}

    class DummyModel:
        def transcribe(self, audio_paths):
            calls["audio_paths"] = audio_paths
            return ["ok"]

    class DummyASRModel:
        @staticmethod
        def from_pretrained(model_name):
            calls["model_name"] = model_name
            return DummyModel()

        @staticmethod
        def restore_from(path):
            calls["restore_from"] = path
            return DummyModel()

    module = types.ModuleType("nemo.collections.asr.models")
    setattr(module, "ASRModel", DummyASRModel)
    monkeypatch.setitem(sys.modules, "nemo", types.ModuleType("nemo"))
    monkeypatch.setitem(
        sys.modules, "nemo.collections", types.ModuleType("nemo.collections")
    )
    monkeypatch.setitem(
        sys.modules, "nemo.collections.asr", types.ModuleType("nemo.collections.asr")
    )
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", module)
    monkeypatch.setattr(nemo_asr, "_audio_duration_seconds", lambda _path: 0.5)

    backend = nemo_asr.NemoASRBackend("local_model.nemo")
    result = backend.transcribe(Path("audio.wav"))

    assert calls["restore_from"] == "local_model.nemo"
    assert calls["audio_paths"] == ["audio.wav"]
    assert result.text == "ok"
    assert result.duration == 0.5


def test_nemo_prepare_audio_passthrough(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"data")

    prepared, temp_path = nemo_asr._prepare_audio_path(audio_path)

    assert prepared == audio_path
    assert temp_path is None


def test_nemo_prepare_audio_converts_non_wav(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"data")
    calls = {}

    def fake_run(cmd, check, stdout, stderr):
        calls["cmd"] = cmd
        return None

    monkeypatch.setattr(nemo_asr.subprocess, "run", fake_run)

    prepared, temp_path = nemo_asr._prepare_audio_path(audio_path)

    assert prepared == audio_path
    assert temp_path is None

    prepared, temp_path = nemo_asr._prepare_audio_path(audio_path, force=True)

    assert prepared.suffix == ".wav"
    assert temp_path == prepared
    assert calls["cmd"][0] == "ffmpeg"
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)


def test_nemo_prepare_audio_converts_stereo_wav(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"data")
    calls = {}

    def fake_run(cmd, check, stdout, stderr):
        calls["cmd"] = cmd
        return None

    monkeypatch.setattr(nemo_asr, "_audio_channel_count", lambda _path: 2)
    monkeypatch.setattr(nemo_asr.subprocess, "run", fake_run)

    prepared, temp_path = nemo_asr._prepare_audio_path(audio_path)

    assert prepared.suffix == ".wav"
    assert temp_path == prepared
    assert calls["cmd"][0] == "ffmpeg"
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)


def test_whisper_cpp_transcribe_scales_segments_and_params(monkeypatch):
    calls = {}

    class DummyModel:
        def __init__(self, model_id, print_progress, print_realtime):
            calls["init"] = (model_id, print_progress, print_realtime)

        def transcribe(self, audio_path, **params):
            calls["audio_path"] = audio_path
            calls["params"] = params
            return [
                SimpleNamespace(text=" hello ", t0=0, t1=100, probability=0.9),
                SimpleNamespace(text="world", t0=100, t1=200, probability=None),
            ]

    model_module = types.ModuleType("pywhispercpp.model")
    setattr(model_module, "Model", DummyModel)
    pkg = types.ModuleType("pywhispercpp")
    monkeypatch.setitem(sys.modules, "pywhispercpp", pkg)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", model_module)

    backend = whisper_cpp.WhisperCppBackend("base.en")
    result = backend.transcribe(
        Path("audio.wav"),
        language="fr",
        temperature=0.4,
        word_timestamps=True,
        prompt="hint",
    )

    assert calls["init"] == ("base.en", False, False)
    assert calls["audio_path"] == "audio.wav"
    assert calls["params"] == {
        "temperature": 0.4,
        "language": "fr",
        "initial_prompt": "hint",
    }
    assert result.text == "hello world"
    assert result.language == "fr"
    assert result.duration == 2.0
    assert result.words is None
    assert result.segments is not None
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.0
    assert result.segments[0].confidence == 0.9
    assert result.segments[1].confidence is None


def test_kyutai_mlx_pad_audio_uses_stt_config():
    backend = kyutai_mlx.KyutaiMlxBackend.__new__(kyutai_mlx.KyutaiMlxBackend)

    class DummyNumpy:
        def __init__(self):
            self.called = None

        def pad(self, audio, pad_width, mode):
            self.called = (audio, pad_width, mode)
            return "padded"

    dummy_np = DummyNumpy()
    setattr(backend, "_np", dummy_np)
    setattr(
        backend,
        "_stt_config",
        {
            "audio_delay_seconds": 0.5,
            "audio_silence_prefix_seconds": 0.25,
        },
    )

    result = kyutai_mlx.KyutaiMlxBackend._pad_audio(backend, "audio")

    assert result == "padded"
    assert dummy_np.called == ("audio", [(0, 0), (6000, 36000)], "constant")


def test_kyutai_mlx_transcribe_returns_empty_for_short_audio():
    backend = kyutai_mlx.KyutaiMlxBackend.__new__(kyutai_mlx.KyutaiMlxBackend)

    class DummyAudio:
        def __init__(self, length):
            self.shape = (1, length)

    calls = {}

    class DummySphn:
        def read(self, path, sample_rate):
            calls["path"] = path
            calls["sample_rate"] = sample_rate
            return DummyAudio(1000), None

    setattr(backend, "_sphn", DummySphn())
    setattr(backend, "_stt_config", None)

    result = backend.transcribe(
        Path("audio.wav"),
        language="en",
        temperature=0.8,
        word_timestamps=True,
        prompt="ignored",
    )

    assert calls["path"] == "audio.wav"
    assert calls["sample_rate"] == 24000
    assert result.text == ""
    assert result.language == "en"
    assert result.duration == pytest.approx(1000 / 24000.0)


def test_kyutai_mlx_pad_audio_returns_input_without_config():
    backend = kyutai_mlx.KyutaiMlxBackend.__new__(kyutai_mlx.KyutaiMlxBackend)
    setattr(backend, "_stt_config", None)
    audio = object()

    result = kyutai_mlx.KyutaiMlxBackend._pad_audio(backend, audio)

    assert result is audio


def test_kyutai_mlx_transcribe_decodes_tokens():
    backend = kyutai_mlx.KyutaiMlxBackend.__new__(kyutai_mlx.KyutaiMlxBackend)
    encode_calls = []
    sampler_calls = []
    step_inputs = []
    gen_calls = {}
    read_calls = {}
    pad_calls = {}

    class DummyAudio:
        def __init__(self, length):
            self.shape = (1, length)

        def __getitem__(self, _item):
            return DummyPcm()

    class DummyPcm:
        def __getitem__(self, _item):
            return "pcm"

    class DummySphn:
        def read(self, path, sample_rate):
            read_calls["path"] = path
            read_calls["sample_rate"] = sample_rate
            return DummyAudio(7680), None

    def pad_audio(audio):
        pad_calls["audio"] = audio
        return audio

    class DummySampler:
        def __init__(self, top_k, temp):
            sampler_calls.append((top_k, temp))

    token_values = iter([4, 0, 3, 6])

    class DummyLmGen:
        def __init__(
            self, model, max_steps, text_sampler, audio_sampler, cfg_coef, check
        ):
            gen_calls["model"] = model
            gen_calls["max_steps"] = max_steps
            gen_calls["cfg_coef"] = cfg_coef
            gen_calls["check"] = check

        def step(self, other_audio_tokens, condition_tensor):
            step_inputs.append((other_audio_tokens, condition_tensor))
            value = next(token_values)
            return [SimpleNamespace(item=lambda: value)]

    class DummyAudioTokenizer:
        def encode_step(self, data):
            encode_calls.append(data)
            return "encoded"

    class DummyTextTokenizer:
        def id_to_piece(self, token):
            mapping = {4: "\u2581hello", 6: "\u2581there"}
            return mapping[token]

    class DummyMxArray:
        def __init__(self, payload):
            self.payload = payload

        def transpose(self, *_args):
            return self

        def __getitem__(self, item):
            if isinstance(item, tuple):
                return self
            if item == 0:
                return self.payload
            return self

    class DummyMx:
        def array(self, payload):
            return DummyMxArray(payload)

    setattr(backend, "_sphn", DummySphn())
    setattr(backend, "_pad_audio", pad_audio)
    setattr(backend, "_utils", SimpleNamespace(Sampler=DummySampler))
    setattr(backend, "_models", SimpleNamespace(LmGen=DummyLmGen))
    setattr(backend, "_audio_tokenizer", DummyAudioTokenizer())
    setattr(backend, "_mx", DummyMx())
    setattr(backend, "_text_tokenizer", DummyTextTokenizer())
    setattr(backend, "_other_codebooks", 1)
    setattr(backend, "_condition_tensor", "cond")
    setattr(backend, "_model", "model")

    result = backend.transcribe(
        Path("audio.wav"),
        language="en",
        temperature=0.2,
        word_timestamps=True,
        prompt="ignored",
    )

    assert read_calls["path"] == "audio.wav"
    assert read_calls["sample_rate"] == 24000
    assert pad_calls["audio"].shape == (1, 7680)
    assert sampler_calls == [(25, 0.2), (250, 0.2)]
    assert gen_calls["max_steps"] == 4
    assert gen_calls["cfg_coef"] == 1.0
    assert gen_calls["check"] is False
    assert step_inputs[0] == ("encoded", "cond")
    assert encode_calls == ["pcm", "pcm", "pcm", "pcm"]
    assert result.text == "hello there"
    assert result.language == "en"
    assert result.duration == pytest.approx(7680 / 24000.0)
    assert result.words is None
    assert result.segments is None


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
