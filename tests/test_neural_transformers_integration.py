"""Opt-in live-model acceptance test for the optional Transformers extra."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from rai.neural.artifacts import LensRegistry
from rai.neural.contracts import EventType, GenerationRequest
from rai.neural.engines.transformers import TransformersEngine
from rai.neural.service import NeuralService


@pytest.mark.asyncio
async def test_pinned_local_model_completes_and_unloads(tmp_path: Path) -> None:
    """Prove the real engine when CI supplies a cached model and exact revision."""
    model_id = os.environ.get("RAI_NEURAL_TEST_MODEL")
    revision = os.environ.get("RAI_NEURAL_TEST_MODEL_REVISION")
    if not model_id or not revision:
        pytest.skip(
            "set RAI_NEURAL_TEST_MODEL and RAI_NEURAL_TEST_MODEL_REVISION to a cached model"
        )
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    lens_dir = tmp_path / "identity-lens"
    lens_dir.mkdir()
    engine = TransformersEngine(
        model_id,
        revision,
        None,
        LensRegistry(tmp_path),
        device="cpu",
        dtype="float32",
    )
    await engine._ensure_loaded()  # pylint: disable=protected-access
    identity = engine._identity()  # pylint: disable=protected-access
    hidden_size = engine._model.config.get_text_config().hidden_size  # pylint: disable=protected-access
    payload = lens_dir / "lens.safetensors"
    safetensors.save_file({"layer.0.jacobian": torch.eye(hidden_size)}, payload)
    manifest = {
        "schema-version": "rai.jlens-artifact.v1",
        "lens-id": "identity-smoke",
        "lens-revision": "test",
        "model": {
            "id": identity.model_id,
            "revision": identity.model_revision,
            "tokenizer-revision": identity.tokenizer_revision,
            "architecture": identity.architecture,
            "dtype": identity.dtype,
            "quantization": identity.quantization,
        },
        "layers": [0],
        "implementation-revision": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "fitting-parameters": {"kind": "identity-smoke"},
        "calibration-corpus": "test",
        "calibration-license": "test-only",
        "artifact-file": payload.name,
        "artifact-sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "created-at": "2026-08-18T00:00:00Z",
    }
    (lens_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    service = NeuralService(engine)

    events = [
        event
        async for event in service.generate(
            GenerationRequest(
                prompt="Hello",
                request_id="live-model-test",
                lens_id="identity-smoke",
                layers=(0,),
                top_k=3,
                max_new_tokens=1,
                timeout_seconds=25,
            )
        )
    ]
    unloaded = await service.unload()

    assert events[-1].event_type is EventType.GENERATION_COMPLETED
    assert any(event.event_type is EventType.NEURAL_STATE_OBSERVED for event in events)
    assert unloaded
    assert not engine.description.loaded
