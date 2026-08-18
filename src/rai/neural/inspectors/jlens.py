"""Safe-tensor backed sparse J-lens readout.

The on-disk tensor format is intentionally narrow: ``layer.<n>.jacobian`` is
the fitted ``d_model x d_model`` average Jacobian.  Readout transports the
residual into the final-layer basis and then uses the loaded model's own final
normalization and unembedding.  This follows the reviewed upstream revision
without accepting its pickle-based checkpoint format at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..artifacts import LensManifest, ModelIdentity
from ..contracts import Concept, NcsiErrorCode, NcsiRuntimeError

MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class Readout:
    """A compact readout suitable for conversion into an NCSI observation."""

    concepts: tuple[Concept, ...]
    reconstruction_error: float
    method: str = "jlens-sparse"


class JLensInspector:
    """Load and evaluate a checksummed, model-compatible sparse lens."""

    def __init__(
        self,
        manifest: LensManifest,
        payload: Path,
        identity: ModelIdentity,
        decode_token: Callable[[int], str],
        unembed: Callable[[object], object],
    ) -> None:
        manifest.ensure_compatible(identity)
        try:
            from safetensors.torch import load_file  # noqa: PLC0415
        except ImportError as exc:
            raise NcsiRuntimeError(
                NcsiErrorCode.MODEL_LOAD_FAILED,
                "the neural extra is required to load J-lens artifacts",
            ) from exc
        try:
            self._tensors: dict[str, Any] = load_file(str(payload), device="cpu")
        except Exception as exc:
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, "unable to load safe J-lens tensors"
            ) from exc
        expected = {f"layer.{layer}.jacobian" for layer in manifest.layers}
        if not expected.issubset(self._tensors):
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, "lens payload is missing declared layers"
            )
        self.manifest = manifest
        self._decode_token = decode_token
        self._unembed = unembed

    def read(self, layer: int, residual: object, top_k: int) -> Readout:
        """Project one last-token residual and return bounded normalized scores."""
        try:
            import torch  # noqa: PLC0415

            jacobian = self._tensors[f"layer.{layer}.jacobian"].to(
                device=residual.device, dtype=residual.dtype
            )
            vector = residual.detach()
            while vector.ndim > 1:
                vector = vector[-1]
            if (
                jacobian.ndim != MATRIX_DIMENSIONS
                or jacobian.shape[0] != vector.shape[0]
                or jacobian.shape[1] != vector.shape[0]
            ):
                raise ValueError("Jacobian and residual dimensions differ")
            transported = torch.mv(jacobian, vector)
            logits = self._unembed(transported)
            probabilities = torch.softmax(logits.float(), dim=-1)
            scores, token_ids = torch.topk(probabilities, min(top_k, probabilities.numel()))
            concepts = tuple(
                Concept(int(token_id), self._decode_token(int(token_id)), float(score))
                for score, token_id in zip(scores.cpu().tolist(), token_ids.cpu().tolist())
            )
            return Readout(concepts=concepts, reconstruction_error=0.0)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, f"J-lens readout failed: {exc}"
            ) from exc
