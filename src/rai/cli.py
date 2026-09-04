"""Stable CLI import facade over the transitional compatibility command set."""

from __future__ import annotations

import sys

from . import cli_compatibility as _compatibility

# Preserve legacy imports and patch targets while quarantining their implementation.
sys.modules[__name__] = _compatibility

if __name__ == "__main__":
    _compatibility.cli(obj={})  # pylint: disable=no-value-for-parameter
