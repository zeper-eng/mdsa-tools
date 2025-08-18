# docs/source/conf.py
import os, sys
sys.path.insert(0, os.path.abspath("../.."))  # adjust if needed

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

# Avoid building native/scientific deps on RTD
autodoc_mock_imports = [
    "numpy", "pandas", "matplotlib", "sklearn",
    "umap", "umap_learn", "mdtraj",
    "python_circos", "pycircos", "requests",
]

html_theme = "sphinx_rtd_theme"
autosummary_generate = True

html_title = "mdsa-tools Documentation"
html_short_title = "mdsa-tools"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # optional extras:
    # "inherited-members": True,
    # "private-members": True,   # include _private
    # "special-members": "__call__",  # dunder methods if you want
}

# If some heavy deps break import at doc build, mock them:
autodoc_mock_imports = ["mdtraj"]  # add others if needed

# Ensure the package can be found
import os, sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../'))

