"""Reject release candidates that do not follow the active release line."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from packaging.version import InvalidVersion, Version

ReleaseKind = Literal["preview", "stable"]


@dataclass(frozen=True)
class ReleaseState:
    """Relevant immutable view of versions already present in Git tags."""

    highest_version: Version | None
    highest_stable: Version | None
    active_prerelease_base: Version | None


def parse_release_tags(tags: Sequence[str]) -> tuple[Version, ...]:
    """Return valid ``vX.Y.Z``-style versions and ignore unrelated tags."""
    versions: list[Version] = []
    for tag in tags:
        value = tag.strip()
        if not value.startswith("v"):
            continue
        try:
            versions.append(Version(value[1:]))
        except InvalidVersion:
            continue
    return tuple(versions)


def release_base(version: Version) -> Version:
    """Return the stable base corresponding to a pre-release version."""
    return Version(f"{version.major}.{version.minor}.{version.micro}")


def release_state(tags: Sequence[str]) -> ReleaseState:
    """Derive the stable floor and any not-yet-promoted pre-release line."""
    versions = parse_release_tags(tags)
    stable_versions = tuple(version for version in versions if not version.is_prerelease)
    highest_stable = max(stable_versions, default=None)
    prerelease_bases = tuple(
        release_base(version)
        for version in versions
        if version.is_prerelease
        and (highest_stable is None or release_base(version) > highest_stable)
    )
    return ReleaseState(
        highest_version=max(versions, default=None),
        highest_stable=highest_stable,
        active_prerelease_base=max(prerelease_bases, default=None),
    )


def validate_candidate(
    kind: ReleaseKind, candidate_text: str, tags: Sequence[str]
) -> tuple[str, ...]:
    """Return all reasons why a calculated candidate must not be released."""
    try:
        candidate = Version(candidate_text.strip())
    except InvalidVersion:
        return (f"semantic-release returned an invalid version: {candidate_text!r}",)

    state = release_state(tags)
    errors: list[str] = []
    if kind == "stable":
        if candidate.is_prerelease:
            errors.append(f"stable job calculated pre-release {candidate}")
        if state.highest_stable is not None and candidate <= state.highest_stable:
            errors.append(
                f"stable candidate {candidate} does not advance existing stable "
                f"{state.highest_stable}"
            )
        if (
            state.active_prerelease_base is not None
            and candidate != state.active_prerelease_base
        ):
            errors.append(
                f"stable candidate {candidate} does not promote active pre-release line "
                f"{state.active_prerelease_base}"
            )
    else:
        if not candidate.is_prerelease:
            errors.append(f"preview job calculated stable version {candidate}")
        if state.highest_version is not None and candidate <= state.highest_version:
            errors.append(
                f"preview candidate {candidate} does not advance existing release "
                f"{state.highest_version}"
            )
        if (
            state.active_prerelease_base is not None
            and release_base(candidate) != state.active_prerelease_base
        ):
            errors.append(
                f"preview candidate {candidate} leaves active pre-release line "
                f"{state.active_prerelease_base}"
            )
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate candidate and tag files produced by the release job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("preview", "stable"))
    parser.add_argument("candidate_file", type=Path)
    parser.add_argument("tags_file", type=Path)
    arguments = parser.parse_args(argv)
    candidate = arguments.candidate_file.read_text(encoding="utf-8").strip()
    tags = arguments.tags_file.read_text(encoding="utf-8").splitlines()
    errors = validate_candidate(arguments.kind, candidate, tags)
    if errors:
        for error in errors:
            print(f"release rejected: {error}")
        return 1
    print(f"release accepted: {arguments.kind} candidate {candidate}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CI entry point
    raise SystemExit(main())
