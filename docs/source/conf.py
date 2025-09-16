exclude_patterns = ["_api/*"]

# docs/source/conf.py
import os, sys
sys.path.insert(0, os.path.abspath(".."))  # so 'mdsa_tools' resolves if installed by RTD

project = "mdsa-tools"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "mdtraj", "matplotlib", "seaborn", "sklearn", "umap", "pandas", "scipy",
    "pycircos",   
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

html_theme = "sphinx_rtd_theme"

