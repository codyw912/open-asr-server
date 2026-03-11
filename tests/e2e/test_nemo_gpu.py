from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.e2e_gpu
def test_nemo_backend_transcribes_sample_on_gpu():
    if os.getenv("OPEN_ASR_RUN_GPU_E2E") != "1":
        pytest.skip("Set OPEN_ASR_RUN_GPU_E2E=1 to run GPU E2E tests")

    import torch  # type: ignore[import-not-found]

    assert torch.cuda.is_available(), "CUDA is not available on this runner"

    from open_asr_server.backends.nemo_asr import NemoASRBackend

    model_id = os.getenv("OPEN_ASR_NEMO_E2E_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    sample_path = Path(
        os.getenv("OPEN_ASR_E2E_AUDIO", "samples/jfk_0_5.flac")
    ).resolve()
    assert sample_path.exists(), f"Sample audio missing: {sample_path}"

    backend = NemoASRBackend(model_id=model_id)
    result = backend.transcribe(sample_path)

    assert result.text.strip() != ""
    assert result.duration > 0
