# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../../src"))

# Copy README.md so it can be included in Sphinx
readme_path = os.path.abspath("../../../README.md")
with open(readme_path, encoding="utf-8") as f:
    readme_content = f.read()

# Fix paths for Sphinx
readme_content = readme_content.replace("docs/sphinx/source/charts_gallery.jpg", "charts_gallery.jpg")
readme_content = readme_content.replace("docs/api/index.md", "api/index.md")

with open(os.path.abspath("README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyPerfAnalytics"
copyright = "2026, Contributors"
author = "Contributors"

version = "1.1.0"
release = "1.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "html_image",
    "html_admonition",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
