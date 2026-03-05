"""Dependency smoke checks for CPU backend extras.

This file is invoked explicitly by CI backend profile jobs.
It is intentionally not matched by default pytest discovery.
"""


def test_import_faster_whisper_dependency():
    import faster_whisper  # type: ignore[import-not-found]

    assert hasattr(faster_whisper, "__version__")


def test_import_whisper_cpp_dependency():
    from pywhispercpp.model import Model  # type: ignore[import-not-found]

    assert callable(Model)
