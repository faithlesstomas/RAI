"""Optional NCSI neural sidecar.

The package deliberately imports no model framework.  Transformers and Torch
are loaded only after the separately started sidecar receives work.
"""

from .contracts import NCSI_SCHEMA_VERSION

__all__ = ["NCSI_SCHEMA_VERSION"]
