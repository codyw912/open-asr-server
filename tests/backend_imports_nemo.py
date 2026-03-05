"""Dependency smoke checks for NeMo backend extras.

This file is invoked explicitly by CI backend profile jobs.
It is intentionally not matched by default pytest discovery.
"""


def test_import_nemo_asr_dependency():
    from nemo.collections.asr.models import ASRModel  # type: ignore[import-not-found]

    assert callable(getattr(ASRModel, "from_pretrained", None))


def test_import_torch_stack_dependencies():
    import torch  # type: ignore[import-not-found]
    import torchaudio  # type: ignore[import-not-found]
    import torchvision  # type: ignore[import-not-found]

    assert hasattr(torch, "__version__")
    assert hasattr(torchaudio, "__version__")
    assert hasattr(torchvision, "__version__")
