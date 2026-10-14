"""Latent space probe integration for cognitive workspace deliberation."""

from __future__ import annotations

from typing import Any

from returns.result import Failure, Result, Success

from rai.assistant.cognitive.models import WorkspaceContribution, WorkspaceState
from rai.kernel.records import ActionFailure, ProducerIdentity, _new_id, _utc_now

PRODUCER = ProducerIdentity(producer_id="jlens-probe", kind="probe", version="1.0.0")


class JLensProbe:
    """Monitors latent semantic drift during workspace deliberation cycles."""

    def __init__(
        self,
        inspector: object = None,
        divergence_threshold: float = 0.85,
    ) -> None:
        self.inspector = inspector
        self.divergence_threshold = divergence_threshold

    @property
    def is_available(self) -> bool:
        """Return True only if a valid neural inspector or J-lens backend is attached."""
        return self.inspector is not None

    async def evaluate_drift(
        self,
        workspace: WorkspaceState,
        latest_contribution: WorkspaceContribution,
    ) -> Result[float, ActionFailure]:
        """Compute semantic drift / divergence between cycle focus and goal representation."""
        if not self.is_available:
            return Success(0.0)

        try:
            # If a callable or inspector is attached, invoke its inspection hook
            if callable(getattr(self.inspector, "inspect_drift", None)):
                drift_score = float(self.inspector.inspect_drift(workspace.goal, latest_contribution.content))
                return Success(drift_score)

            # Fallback zero-drift if inspector is passive
            return Success(0.0)
        except Exception as e:
            return Failure(
                ActionFailure(
                    record_id=_new_id(),
                    timestamp=_utc_now(),
                    producer=PRODUCER,
                    request_id=workspace.turn_id,
                    capability="jlens_probe",
                    code="PROBE_EVALUATION_FAILED",
                    message=f"Failed to evaluate latent semantic drift: {e}",
                )
            )

    def is_divergent(self, drift_score: float) -> bool:
        """Check if drift exceeds divergence threshold."""
        return self.is_available and drift_score >= self.divergence_threshold
