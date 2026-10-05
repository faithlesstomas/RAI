import os
import sys
import atexit
import tempfile

# Autodoc imports the ASGI module. Never read or mutate a developer's settings,
# databases or credentials while building documentation.
_docs_runtime = tempfile.TemporaryDirectory(prefix="rai-sphinx-")
atexit.register(_docs_runtime.cleanup)
for _kind in ("CONFIG", "DATA", "CACHE", "RUNTIME", "STATE"):
    os.environ[f"RAI_{_kind}_DIR"] = os.path.join(_docs_runtime.name, _kind.lower())
os.environ.pop("RAI_CONFIG_FILE", None)

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('../src'))

import rai

# -- Project information -----------------------------------------------------
project = 'RAI'
copyright = '2026, RAI Team'
author = 'RAI Team'
version = rai.__version__
release = rai.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_title = "Rich AI - Secure AI Integration for Linux"
