"""Checksummed J-lens artifact admission tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rai.neural.artifacts import LensManifest, ModelIdentity
from rai.neural.contracts import NcsiErrorCode, NcsiRuntimeError
from rai.neural.inspectors.jlens import JLensInspector


def _write_artifact(directory: Path, **model_overrides: object) -> LensManifest:
    directory.mkdir()
    payload = directory / "lens.safetensors"
    payload.write_bytes(b"safe tensor placeholder")
    model = {
        "id": "model",
        "revision": "model-revision",
        "tokenizer-revision": "tokenizer-revision",
        "architecture": "FixtureForCausalLM",
        "dtype": "float32",
        "quantization": None,
    }
    model.update(model_overrides)
    manifest = {
        "schema-version": "rai.jlens-artifact.v1",
        "lens-id": "lens",
        "lens-revision": "lens-revision",
        "model": model,
        "layers": [1, 2],
        "implementation-revision": "implementation-revision",
        "fitting-parameters": {"rank": 8},
        "calibration-corpus": "corpus",
        "calibration-license": "MIT",
        "artifact-file": payload.name,
        "artifact-sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "created-at": "2026-08-18T00:00:00Z",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return LensManifest.load(directory)


def test_loads_checksummed_compatible_manifest(tmp_path: Path) -> None:
    manifest = _write_artifact(tmp_path / "lens")
    identity = ModelIdentity(
        "model",
        "model-revision",
        "tokenizer-revision",
        "FixtureForCausalLM",
        "float32",
    )

    manifest.ensure_compatible(identity)

    assert manifest.to_public_dict()["layers"] == [1, 2]
    assert "artifact-file" not in manifest.to_public_dict()


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "lens"
    _write_artifact(directory)
    (directory / "lens.safetensors").write_bytes(b"tampered")

    with pytest.raises(NcsiRuntimeError) as raised:
        LensManifest.load(directory)

    assert raised.value.code is NcsiErrorCode.LENS_INCOMPATIBLE


def test_rejects_model_revision_mismatch(tmp_path: Path) -> None:
    manifest = _write_artifact(tmp_path / "lens")
    incompatible = ModelIdentity(
        "model", "other-revision", "tokenizer-revision", "FixtureForCausalLM", "float32"
    )

    with pytest.raises(NcsiRuntimeError) as raised:
        manifest.ensure_compatible(incompatible)

    assert raised.value.code is NcsiErrorCode.LENS_INCOMPATIBLE


def test_safe_tensor_lens_produces_bounded_readout(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    directory = tmp_path / "lens"
    directory.mkdir()
    payload = directory / "lens.safetensors"
    safetensors.save_file({"layer.1.jacobian": torch.eye(3)}, payload)
    manifest_data = {
        "schema-version": "rai.jlens-artifact.v1",
        "lens-id": "lens",
        "lens-revision": "lens-revision",
        "model": {
            "id": "model",
            "revision": "model-revision",
            "tokenizer-revision": "tokenizer-revision",
            "architecture": "FixtureForCausalLM",
            "dtype": "float32",
            "quantization": None,
        },
        "layers": [1],
        "implementation-revision": "implementation-revision",
        "fitting-parameters": {},
        "calibration-corpus": "corpus",
        "calibration-license": "MIT",
        "artifact-file": payload.name,
        "artifact-sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "created-at": "2026-08-18T00:00:00Z",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    identity = ModelIdentity(
        "model",
        "model-revision",
        "tokenizer-revision",
        "FixtureForCausalLM",
        "float32",
    )
    inspector = JLensInspector(
        LensManifest.load(directory),
        payload,
        identity,
        lambda token_id: f"token-{token_id}",
        lambda residual: residual,
    )

    readout = inspector.read(1, torch.tensor([0.0, 2.0, 0.0]), top_k=2)

    assert len(readout.concepts) == 2  # noqa: PLR2004
    assert readout.concepts[0].token_id == 1
    assert 0.0 <= readout.concepts[0].score <= 1.0
