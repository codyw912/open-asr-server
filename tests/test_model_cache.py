import sys
import types

from open_asr_server.utils.model_cache import get_model_cache_dir, resolve_model_path


def test_get_model_cache_dir_unset(monkeypatch):
    monkeypatch.delenv("OPEN_ASR_SERVER_MODEL_DIR", raising=False)
    assert get_model_cache_dir() is None


def test_get_model_cache_dir_set(monkeypatch, tmp_path):
    monkeypatch.setenv("OPEN_ASR_SERVER_MODEL_DIR", str(tmp_path))
    assert get_model_cache_dir() == tmp_path


def test_resolve_model_path_prefers_local(tmp_path):
    local = tmp_path / "model"
    local.mkdir()
    assert resolve_model_path(str(local)) == local


def test_resolve_model_path_uses_snapshot_download(monkeypatch, tmp_path):
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, cache_dir=None):
        assert repo_id == "repo-id"
        assert cache_dir is None
        return str(tmp_path / "cache" / repo_id)

    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    resolved = resolve_model_path("repo-id")
    assert resolved == tmp_path / "cache" / "repo-id"


def test_resolve_model_path_uses_cache_dir(monkeypatch, tmp_path):
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, cache_dir=None):
        assert repo_id == "repo-id"
        assert cache_dir == str(tmp_path)
        return str(tmp_path / "cache" / repo_id)

    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("OPEN_ASR_SERVER_MODEL_DIR", str(tmp_path))

    resolved = resolve_model_path("repo-id")
    assert resolved == tmp_path / "cache" / "repo-id"
