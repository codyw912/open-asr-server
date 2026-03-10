from open_asr_server import install_hints


def test_backend_runtime_status_reports_platform_incompatibility():
    status, reason = install_hints.backend_runtime_status(
        "parakeet-mlx",
        platform_name="linux",
        python_version="3.11",
        has_nvidia=False,
    )

    assert status == "platform_incompatible"
    assert reason == "supported platforms: darwin"


def test_backend_runtime_status_reports_python_incompatibility():
    status, reason = install_hints.backend_runtime_status(
        "faster-whisper",
        platform_name="linux",
        python_version="3.14",
        has_nvidia=False,
    )

    assert status == "python_incompatible"
    assert reason == "supported python: 3.11, 3.12, 3.13"


def test_backend_runtime_status_reports_gpu_requirement():
    status, reason = install_hints.backend_runtime_status(
        "nemo-parakeet",
        platform_name="linux",
        python_version="3.11",
        has_nvidia=False,
    )

    assert status == "requires_gpu"
    assert reason == "requires NVIDIA GPU"


def test_recommended_python_for_cpu_and_cuda_extras():
    assert install_hints.recommended_python_for_extra("cpu") == "3.11"
    assert install_hints.recommended_python_for_extra("nemo") == "3.11"
