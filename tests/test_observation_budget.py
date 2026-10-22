"""Tests for ObservationBudgeter and structural observation rendering."""
# ruff: noqa: PLR2004

from __future__ import annotations

import json
from pathlib import Path
import pytest

from rai.assistant.observation_budget import (
    render_observation,
    render_observation_payload,
)
from rai.assistant.records import (
    ObservationArtifact,
    ToolObservation,
    summarize_observation_schema,
)
from rai.kernel.records import DataClass, ProducerIdentity

PRODUCER = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")


def test_render_observation_within_budget() -> None:
    content = {
        "status": "SUCCEEDED",
        "items": [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}],
    }
    rendered = render_observation(content, budget_chars=2000)
    assert not rendered.is_reduced
    assert rendered.shown_count == 2
    assert rendered.total_count == 2
    assert rendered.preview_footer is None
    parsed = json.loads(rendered.rendered_text)
    assert len(parsed["items"]) == 2


def test_render_observation_strips_technical_fields() -> None:
    content = {
        "handle": "h_123",
        "fingerprint": "fp_abc",
        "expires_at": "2026-10-22T00:00:00Z",
        "applications": [
            {"name": "App 1", "handle": "h_sub", "desktop_id": "app1.desktop"}
        ],
    }
    rendered = render_observation(content, budget_chars=1000)
    parsed = json.loads(rendered.rendered_text)
    assert "handle" not in parsed
    assert "fingerprint" not in parsed
    assert "expires_at" not in parsed
    assert "handle" not in parsed["applications"][0]
    assert parsed["applications"][0]["name"] == "App 1"


def test_render_observation_with_123_applications() -> None:
    """Acceptance criterion: 123 applications, valid JSON, shown/total, no '[truncated N chars]'."""
    apps = [
        {
            "name": f"App_{i}",
            "desktop_id": f"org.example.app_{i}.desktop",
            "categories": ["Utility", "Video" if i % 10 == 0 else "Office"],
            "comment": f"Sample application description number {i}",
            "generic_name": f"Generic {i}",
            "handle": f"handle_{i}",
            "expires_at": "2026-10-22T00:00:00Z",
        }
        for i in range(1, 124)
    ]
    raw_content = {
        "total_count": len(apps),
        "applications": apps,
    }

    artifact = ObservationArtifact(
        artifact_id="obs_1",
        session_id="session-test",
        turn_id="turn-test",
        capability="application.list",
        request_id="req-test",
        data_class=DataClass.LOCAL,
        content=raw_content,
        schema_summary=summarize_observation_schema(raw_content),
        size_chars=len(json.dumps(raw_content)),
        producer=PRODUCER,
    )

    # Render with a typical prompt observation budget (e.g. 3000 chars)
    budget = 3000
    rendered = render_observation(artifact, budget_chars=budget)

    # 1. Reduction happened
    assert rendered.is_reduced
    assert rendered.total_count == 123
    assert rendered.shown_count is not None
    assert 0 < rendered.shown_count < 123

    # 2. Output is ALWAYS valid JSON
    parsed = json.loads(rendered.rendered_text)
    assert isinstance(parsed, dict)
    assert "applications" in parsed
    assert len(parsed["applications"]) == rendered.shown_count

    # 3. Output contains _meta with shown and total
    assert "_meta" in parsed
    assert parsed["_meta"]["shown"] == rendered.shown_count
    assert parsed["_meta"]["total"] == 123

    # 4. CRITICAL: Model never sees "[truncated N chars]" string
    assert "[truncated " not in rendered.rendered_text
    assert "chars]" not in rendered.rendered_text

    # 5. Preview footer affordance provides handle and operations
    assert rendered.preview_footer is not None
    assert "obs_1: 123 records" in rendered.preview_footer
    assert f"{rendered.shown_count} shown" in rendered.preview_footer
    assert "observation.search/filter on obs_1" in rendered.preview_footer


def test_render_observation_schema_only_fallback() -> None:
    """When budget is extremely tight, provide schema-only preview."""
    apps = [{"name": f"App_{i}", "desc": "x" * 500} for i in range(50)]
    content = {"applications": apps, "total_count": 50}
    artifact = ObservationArtifact(
        artifact_id="obs_tight",
        session_id="session-test",
        turn_id="turn-test",
        capability="application.list",
        request_id="req-test",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=25000,
        producer=PRODUCER,
    )

    # Very small budget of 250 chars
    rendered = render_observation(artifact, budget_chars=250)
    assert rendered.is_reduced
    parsed = json.loads(rendered.rendered_text)
    assert "_schema_preview" in parsed
    assert parsed["_schema_preview"]["total_count"] == 50
    assert "obs_tight" in rendered.preview_footer


def test_render_observation_payload_string_length_cap() -> None:
    long_str = "a" * 1000
    res = render_observation_payload(long_str, budget_chars=100)
    assert len(res) <= 150
    assert "[truncated" in res
