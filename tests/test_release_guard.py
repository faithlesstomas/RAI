"""Regression coverage for release-line validation."""

from ci.release_guard import release_state, validate_candidate
from packaging.version import Version


def test_release_state_finds_unpromoted_prerelease_line() -> None:
    state = release_state(("v0.3.1", "v0.4.0-alpha.4", "v0.5.0-alpha.1"))

    assert state.highest_stable == Version("0.3.1")
    assert state.active_prerelease_base == Version("0.5.0")


def test_stable_candidate_must_promote_active_prerelease_line() -> None:
    errors = validate_candidate(
        "stable",
        "0.4.0",
        ("v0.3.1", "v0.4.0-alpha.4", "v0.5.0-alpha.1"),
    )

    assert errors == (
        "stable candidate 0.4.0 does not promote active pre-release line 0.5.0",
    )


def test_stable_candidate_accepts_exact_prerelease_promotion() -> None:
    assert not validate_candidate(
        "stable",
        "0.5.0",
        ("v0.3.1", "v0.4.0-alpha.4", "v0.5.0-alpha.1"),
    )


def test_stable_candidate_must_advance_existing_stable_release() -> None:
    errors = validate_candidate("stable", "0.4.0", ("v0.3.1", "v0.4.0"))

    assert errors == (
        "stable candidate 0.4.0 does not advance existing stable 0.4.0",
    )


def test_preview_advances_existing_active_line() -> None:
    tags = ("v0.4.0", "v0.5.0-alpha.1")

    assert not validate_candidate("preview", "0.5.0-alpha.2", tags)
    assert validate_candidate("preview", "0.6.0-alpha.1", tags) == (
        "preview candidate 0.6.0a1 leaves active pre-release line 0.5.0",
    )


def test_preview_after_stable_starts_calculated_next_line() -> None:
    assert not validate_candidate("preview", "0.6.0-alpha.1", ("v0.5.0",))


def test_unrelated_and_invalid_tags_are_ignored() -> None:
    state = release_state(("nightly", "vnot-a-version", "v0.5.0"))

    assert state.highest_version == Version("0.5.0")
    assert state.active_prerelease_base is None
