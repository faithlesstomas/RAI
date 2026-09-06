"""Opt-in metadata-only filesystem observation for approved roots."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

from .common import emit

IGNORED_DIRECTORIES = frozenset({".git", ".hg", ".svn", "__pycache__", ".venv", "node_modules"})
MIN_INTERVAL_SECONDS = 0.1


def approved_roots(values: list[str]) -> tuple[Path, ...]:
    roots = tuple(Path(value).expanduser().resolve(strict=True) for value in values)
    if not roots:
        raise ValueError("at least one approved root is required")
    if any(root == Path(root.anchor) for root in roots):
        raise ValueError("filesystem root cannot be monitored")
    return roots


def metadata_snapshot(roots: tuple[Path, ...]) -> dict[Path, int]:
    """Read names and mtimes only; file content is never opened."""
    snapshot: dict[Path, int] = {}
    for root in roots:
        for current, directories, files in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if name not in IGNORED_DIRECTORIES]
            directory = Path(current)
            for name in files:
                path = directory / name
                try:
                    if path.is_symlink():
                        continue
                    snapshot[path] = path.stat().st_mtime_ns
                except OSError:
                    continue
    return snapshot


def changed_paths(
    previous: dict[Path, int], current: dict[Path, int]
) -> Iterator[tuple[str, Path]]:
    for path in sorted(current.keys() - previous.keys()):
        yield "created", path
    for path in sorted(previous.keys() - current.keys()):
        yield "deleted", path
    for path in sorted(previous.keys() & current.keys()):
        if previous[path] != current[path]:
            yield "save", path


def _emit(kind: str, path: Path) -> None:
    lowered_parts = {part.casefold() for part in path.parts}
    if kind != "deleted" and lowered_parts & {"build", "dist", "target"}:
        kind = "build"
    elif kind != "deleted" and path.name.casefold() in {
        "junit.xml", "test-results.xml", ".coverage"
    }:
        kind = "test_run"
    emit("filesystem", kind, path=str(path), resource_id=f"file:{path}")


async def watch(roots: tuple[Path, ...], interval: float, *, once: bool = False) -> None:
    previous: dict[Path, int] = {}
    while True:
        current = metadata_snapshot(roots)
        for kind, path in changed_paths(previous, current):
            _emit(kind, path)
        if once:
            return
        previous = current
        await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.interval < MIN_INTERVAL_SECONDS:
        parser.error("interval must be at least 0.1 seconds")
    try:
        roots = approved_roots(arguments.roots)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    asyncio.run(watch(roots, arguments.interval, once=arguments.once))


if __name__ == "__main__":  # pragma: no cover
    main()
