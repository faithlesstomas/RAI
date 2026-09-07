"""Install the bundled GNOME Shell extension for the current user."""

from __future__ import annotations

import os
import shutil
from importlib.resources import files
from pathlib import Path

EXTENSION_UUID = "rai-history@tk-lab1"


def extension_root() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    return base / "gnome-shell" / "extensions" / EXTENSION_UUID


def install(destination: Path | None = None) -> Path:
    target = destination or extension_root()
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    assets = files("rai.history").joinpath("gnome_extension")
    for name in ("extension.js", "metadata.json"):
        with assets.joinpath(name).open("rb") as source:
            with (target / name).open("wb") as output:
                shutil.copyfileobj(source, output)
        (target / name).chmod(0o600)
    return target


def main() -> None:
    target = install()
    print(target)
    print(f"Enable with: gnome-extensions enable {EXTENSION_UUID}")


if __name__ == "__main__":  # pragma: no cover
    main()
