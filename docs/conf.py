from importlib.metadata import version

project = "pylibfreenect3"
author = "pylibfreenect3 contributors"
release = version("pylibfreenect3")
version = ".".join(release.split(".")[:2])
extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]
autosummary_generate = True
html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build"]
