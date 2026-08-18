"""Versioned, checksummed J-lens artifact manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import NcsiErrorCode, NcsiRuntimeError

ARTIFACT_SCHEMA_VERSION = "rai.jlens-artifact.v1"
MANIFEST_NAME = "manifest.json"
SHA256_HEX_LENGTH = 64


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise NcsiRuntimeError(
            NcsiErrorCode.LENS_INCOMPATIBLE, f"artifact manifest has invalid {key}"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for block in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    """Exact model/tokenizer identity used for artifact admission."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    architecture: str
    dtype: str
    quantization: str | None = None


@dataclass(frozen=True)
class LensManifest:
    """Reviewed metadata for one immutable J-lens payload."""

    lens_id: str
    lens_revision: str
    model: ModelIdentity
    layers: tuple[int, ...]
    implementation_revision: str
    fitting_parameters: Mapping[str, Any]
    calibration_corpus: str
    calibration_license: str
    artifact_file: str
    artifact_sha256: str
    created_at: str
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    @classmethod
    def load(cls, directory: Path) -> "LensManifest":
        manifest_path = directory / MANIFEST_NAME
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_NOT_FOUND, f"lens manifest not found: {manifest_path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, "lens manifest is not readable JSON"
            ) from exc
        if not isinstance(data, dict) or data.get("schema-version") != ARTIFACT_SCHEMA_VERSION:
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, "unsupported lens artifact schema"
            )
        model_data = data.get("model")
        if not isinstance(model_data, dict):
            raise NcsiRuntimeError(NcsiErrorCode.LENS_INCOMPATIBLE, "model identity is missing")
        raw_layers = data.get("layers")
        if (
            not isinstance(raw_layers, list)
            or not raw_layers
            or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in raw_layers)
        ):
            raise NcsiRuntimeError(NcsiErrorCode.LENS_INCOMPATIBLE, "layers are invalid")
        fitting = data.get("fitting-parameters")
        if not isinstance(fitting, dict):
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, "fitting-parameters must be an object"
            )
        manifest = cls(
            schema_version=data["schema-version"],
            lens_id=_required_string(data, "lens-id"),
            lens_revision=_required_string(data, "lens-revision"),
            model=ModelIdentity(
                model_id=_required_string(model_data, "id"),
                model_revision=_required_string(model_data, "revision"),
                tokenizer_revision=_required_string(model_data, "tokenizer-revision"),
                architecture=_required_string(model_data, "architecture"),
                dtype=_required_string(model_data, "dtype"),
                quantization=model_data.get("quantization"),
            ),
            layers=tuple(raw_layers),
            implementation_revision=_required_string(data, "implementation-revision"),
            fitting_parameters=fitting,
            calibration_corpus=_required_string(data, "calibration-corpus"),
            calibration_license=_required_string(data, "calibration-license"),
            artifact_file=_required_string(data, "artifact-file"),
            artifact_sha256=_required_string(data, "artifact-sha256"),
            created_at=_required_string(data, "created-at"),
        )
        manifest.validate_payload(directory)
        return manifest

    def validate_payload(self, directory: Path) -> Path:
        payload = (directory / self.artifact_file).resolve()
        try:
            payload.relative_to(directory.resolve())
        except ValueError as exc:
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, "artifact-file escapes its lens directory"
            ) from exc
        if not payload.is_file():
            raise NcsiRuntimeError(NcsiErrorCode.LENS_NOT_FOUND, "lens payload is missing")
        if (
            len(self.artifact_sha256) != SHA256_HEX_LENGTH
            or _sha256(payload) != self.artifact_sha256.lower()
        ):
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, "lens payload checksum does not match"
            )
        return payload

    def ensure_compatible(self, identity: ModelIdentity) -> None:
        """Reject every undeclared checkpoint, tokenizer, or execution change."""
        fields = (
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "architecture",
            "dtype",
            "quantization",
        )
        mismatches = [name for name in fields if getattr(self.model, name) != getattr(identity, name)]
        if mismatches:
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE,
                "lens/model identity mismatch: " + ", ".join(mismatches),
            )

    def to_public_dict(self) -> dict[str, object]:
        """Return discovery metadata without exposing local filesystem paths."""
        return {
            "lens-id": self.lens_id,
            "lens-revision": self.lens_revision,
            "model-id": self.model.model_id,
            "model-revision": self.model.model_revision,
            "tokenizer-revision": self.model.tokenizer_revision,
            "layers": list(self.layers),
            "readout-method": "jlens-sparse",
        }


class LensRegistry:
    """Discover immutable lens directories below an XDG data root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def manifests(self) -> tuple[LensManifest, ...]:
        if not self.root.is_dir():
            return ()
        manifests: list[LensManifest] = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and (path / MANIFEST_NAME).is_file():
                manifests.append(LensManifest.load(path))
        return tuple(manifests)

    def get(self, lens_id: str) -> tuple[LensManifest, Path]:
        for path in self.root.iterdir() if self.root.is_dir() else ():
            if path.is_dir() and (path / MANIFEST_NAME).is_file():
                manifest = LensManifest.load(path)
                if manifest.lens_id == lens_id:
                    return manifest, manifest.validate_payload(path)
        raise NcsiRuntimeError(NcsiErrorCode.LENS_NOT_FOUND, f"unknown lens: {lens_id}")
