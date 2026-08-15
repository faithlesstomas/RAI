import os
import sys

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
html_static_path = ['_static']
html_title = "Rich AI - A Local-First Agent Runtime for Linux"
