"""Offline fitting and safe packaging for the pinned Jacobian Lens reference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ARTIFACT_SCHEMA_VERSION, LensManifest
from .contracts import NcsiErrorCode, NcsiRuntimeError

JLENS_IMPLEMENTATION_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


@dataclass(frozen=True)
class FitConfig:
    """Reproducible inputs for one offline Jacobian-lens fitting run."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    lens_id: str
    lens_revision: str
    prompts_path: Path
    output_dir: Path
    calibration_corpus: str
    calibration_license: str
    layers: tuple[int, ...] = ()
    device: str = "auto"
    dtype: str = "auto"
    quantization: str | None = None
    dim_batch: int = 8
    max_seq_len: int = 128
    skip_first: int = 16


def _read_prompts(path: Path) -> tuple[str, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NcsiRuntimeError(
            NcsiErrorCode.INVALID_REQUEST, "calibration prompts must be readable JSON"
        ) from exc
    if (
        not isinstance(data, list)
        or not data
        or any(not isinstance(prompt, str) or not prompt for prompt in data)
    ):
        raise NcsiRuntimeError(
            NcsiErrorCode.INVALID_REQUEST, "calibration corpus must be a non-empty string array"
        )
    return tuple(data)


def _load_model(config: FitConfig) -> tuple[Any, Any]:
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise NcsiRuntimeError(
            NcsiErrorCode.MODEL_LOAD_FAILED,
            "install rai[inference-jlens] to fit an artifact",
        ) from exc
    dtype: str | Any = config.dtype
    if config.dtype != "auto":
        dtype = getattr(torch, config.dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, revision=config.tokenizer_revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            dtype=dtype,
            device_map=config.device,
        )
    except Exception as exc:
        raise NcsiRuntimeError(
            NcsiErrorCode.MODEL_LOAD_FAILED, f"unable to load fitting model: {exc}"
        ) from exc
    return model, tokenizer


def _save_tensors(path: Path, jacobians: dict[int, object]) -> None:
    temp_path: str | None = None
    try:
        from safetensors.torch import save_file  # noqa: PLC0415

        tensors = {
            f"layer.{layer}.jacobian": tensor.detach().cpu().contiguous()
            for layer, tensor in jacobians.items()
        }
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".safetensors", delete=False
        ) as temp:
            temp_path = temp.name
        save_file(tensors, temp_path)
        os.replace(temp_path, path)
    except Exception as exc:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        raise NcsiRuntimeError(
            NcsiErrorCode.LENS_INCOMPATIBLE, f"unable to package fitted lens: {exc}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_lens_artifact(config: FitConfig) -> LensManifest:
    """Fit with pinned upstream code and emit a validated, non-pickle artifact."""
    if config.quantization is not None:
        raise NcsiRuntimeError(
            NcsiErrorCode.MODEL_INCOMPATIBLE,
            "quantized J-lens fitting is not implemented; omit quantization",
        )
    prompts = _read_prompts(config.prompts_path)
    existing = set(config.output_dir.iterdir()) if config.output_dir.exists() else set()
    checkpoint_path = config.output_dir / "fit.checkpoint.pt"
    if existing - {checkpoint_path}:
        raise NcsiRuntimeError(
            NcsiErrorCode.INVALID_REQUEST, "output directory must be absent or empty"
        )
    config.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    model, tokenizer = _load_model(config)
    try:
        import jlens  # noqa: PLC0415

        wrapped = jlens.from_hf(model, tokenizer)
        lens = jlens.fit(
            wrapped,
            prompts,
            source_layers=list(config.layers) or None,
            dim_batch=config.dim_batch,
            max_seq_len=config.max_seq_len,
            skip_first=config.skip_first,
            checkpoint_path=str(checkpoint_path),
        )
    except ImportError as exc:
        raise NcsiRuntimeError(
            NcsiErrorCode.MODEL_LOAD_FAILED, "pinned jlens dependency is not installed"
        ) from exc
    except Exception as exc:
        raise NcsiRuntimeError(
            NcsiErrorCode.INFERENCE_FAILED, f"Jacobian-lens fitting failed: {exc}"
        ) from exc
    payload = config.output_dir / "lens.safetensors"
    _save_tensors(payload, lens.jacobians)
    resolved_model_revision = getattr(model.config, "_commit_hash", None) or config.model_revision
    resolved_tokenizer_revision = (
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or config.tokenizer_revision
    )
    architectures = getattr(model.config, "architectures", None) or []
    manifest = {
        "schema-version": ARTIFACT_SCHEMA_VERSION,
        "lens-id": config.lens_id,
        "lens-revision": config.lens_revision,
        "model": {
            "id": config.model_id,
            "revision": resolved_model_revision,
            "tokenizer-revision": resolved_tokenizer_revision,
            "architecture": architectures[0] if architectures else type(model).__name__,
            "dtype": str(model.dtype).removeprefix("torch."),
            "quantization": config.quantization,
        },
        "layers": lens.source_layers,
        "implementation-revision": JLENS_IMPLEMENTATION_REVISION,
        "fitting-parameters": {
            "n-prompts": lens.n_prompts,
            "dim-batch": config.dim_batch,
            "max-seq-len": config.max_seq_len,
            "skip-first": config.skip_first,
        },
        "calibration-corpus": config.calibration_corpus,
        "calibration-license": config.calibration_license,
        "artifact-file": payload.name,
        "artifact-sha256": _sha256(payload),
        "created-at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = config.output_dir / "manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=config.output_dir, suffix=".json", delete=False
    ) as temp:
        json.dump(manifest, temp, indent=2)
        temp.write("\n")
        temp_path = temp.name
    os.replace(temp_path, manifest_path)
    checkpoint_path.unlink(missing_ok=True)
    return LensManifest.load(config.output_dir)
