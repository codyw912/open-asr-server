from __future__ import annotations

from pathlib import Path
import tomllib


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]
    return project["optional-dependencies"]


def test_canonical_backend_extras_exist():
    optional_dependencies = _optional_dependencies()

    assert "parakeet-mlx" in optional_dependencies
    assert "whisper-mlx" in optional_dependencies
    assert "lightning-whisper-mlx" in optional_dependencies
    assert "kyutai-mlx" in optional_dependencies
    assert "faster-whisper" in optional_dependencies
    assert "whisper-cpp" in optional_dependencies
    assert "nemo" in optional_dependencies


def test_hardware_bundle_extras_exist():
    optional_dependencies = _optional_dependencies()

    assert "metal" in optional_dependencies
    assert "cpu" in optional_dependencies
    assert "cuda" in optional_dependencies


def test_legacy_mlx_extra_names_removed():
    optional_dependencies = _optional_dependencies()

    assert "parakeet" not in optional_dependencies
    assert "whisper" not in optional_dependencies
    assert "lightning-whisper" not in optional_dependencies
