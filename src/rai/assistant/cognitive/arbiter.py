"""Salience arbiter selecting winning contribution for global workspace broadcast."""

from __future__ import annotations

from .models import WorkspaceContribution, WorkspaceContributionKind, WorkspaceState


class SalienceArbiter:
    """Selects the most salient contribution to broadcast to all specialists."""

    def __init__(self, convergence_threshold: float = 0.8) -> None:
        self.convergence_threshold = convergence_threshold

    def select_winner(
        self,
        contributions: list[WorkspaceContribution],
        workspace: WorkspaceState,
    ) -> WorkspaceContribution | None:
        """Evaluate contributions and return the winner with highest effective salience."""
        if not contributions:
            return None

        # Filter out invalid or zero-salience items
        valid = [c for c in contributions if c.salience > 0.0]
        if not valid:
            return None

        # Check for active critiques that penalize repeated actions or ungrounded drafts
        critiques = [
            c for c in workspace.history if c.kind == WorkspaceContributionKind.CRITIQUE
        ]

        scored: list[tuple[float, WorkspaceContribution]] = []
        for c in valid:
            effective = c.salience

            # If there's a critique against this specialist, discount it
            for crit in critiques:
                if crit.specialist_name == c.specialist_name:
                    effective *= 0.5

            # If tool call was already repeated with identical arguments, penalize heavily
            if c.kind == WorkspaceContributionKind.TOOL_CALL:
                for past in workspace.history:
                    if past.kind == WorkspaceContributionKind.TOOL_CALL and past.payload == c.payload:
                        effective *= 0.2

            scored.append((effective, c))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
