"""Offline reference fitter packaging tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rai.neural.fitting import FitConfig, JLENS_IMPLEMENTATION_REVISION, fit_lens_artifact


def test_fitter_packages_reference_jacobians_without_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["a sufficiently long calibration prompt"]), encoding="utf-8")
    model = SimpleNamespace(
        config=SimpleNamespace(
            _commit_hash="resolved-model",
            architectures=["FixtureForCausalLM"],
        ),
        dtype=torch.float32,
    )
    tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": "resolved-tokenizer"})
    fake_lens = SimpleNamespace(
        jacobians={0: torch.eye(3)}, source_layers=[0], n_prompts=1
    )
    fake_jlens = SimpleNamespace(
        from_hf=lambda loaded_model, loaded_tokenizer: (loaded_model, loaded_tokenizer),
        fit=lambda *args, **kwargs: fake_lens,
    )
    monkeypatch.setattr("rai.neural.fitting._load_model", lambda config: (model, tokenizer))
    monkeypatch.setitem(sys.modules, "jlens", fake_jlens)
    output = tmp_path / "artifact"

    manifest = fit_lens_artifact(
        FitConfig(
            model_id="fixture-model",
            model_revision="requested-model",
            tokenizer_revision="requested-tokenizer",
            lens_id="fixture-lens",
            lens_revision="fit-1",
            prompts_path=prompts,
            output_dir=output,
            calibration_corpus="fixture-corpus",
            calibration_license="MIT",
            layers=(0,),
        )
    )

    tensors = safetensors.load_file(output / "lens.safetensors")
    assert set(tensors) == {"layer.0.jacobian"}
    assert manifest.model.model_revision == "resolved-model"
    assert manifest.model.tokenizer_revision == "resolved-tokenizer"
    assert manifest.implementation_revision == JLENS_IMPLEMENTATION_REVISION
    assert not (output / "fit.checkpoint.pt").exists()
    assert not list(output.glob("*.pt"))
