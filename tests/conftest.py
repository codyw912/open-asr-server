import pytest


@pytest.fixture(autouse=True)
def reset_open_asr_env(monkeypatch):
    monkeypatch.delenv("OPEN_ASR_SERVER_MODEL_DIR", raising=False)
    monkeypatch.delenv("OPEN_ASR_SERVER_HF_TOKEN", raising=False)
