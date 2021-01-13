# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import warnings

from gtda import __version__

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'giotto-tda'
copyright = '2021, L2F SA'
author = 'Guillaume Tauzin, Umberto Lupo, Matteo Caorsi, Anibal Medina, ' \
         'Lewis Tunstall, Wojciech Reise'

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'numpydoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    #'sphinx.ext.imgconverter',
    'sphinx_issues',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon'
    # 'custom_references_resolver' # custom for sklearn, not sure what it does
]

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = True

# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get('NO_MATHJAX'):
    extensions.append('sphinx.ext.imgmath')
    imgmath_image_format = 'svg'
else:
    extensions.append('sphinx.ext.mathjax')
    mathjax_path = ('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/'
                    'MathJax.js?config=TeX-AMS_SVG')

autodoc_default_options = {'members': True, 'inherited-members': True}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates/']

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '**neural_network**',
    'templates/*.rst',
    'theory/before_glossary.rst'
]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'logo_only': True,
}

# List versions
current_version = os.environ['VERSION']
html_theme_options.update({'current_version': current_version})
try:
    with open('versions', 'r') as f:
        _versions = [c[2:] for c in f.read().splitlines()]
    _versions = list(filter(lambda c: not(c.startswith('.')), _versions))
except FileNotFoundError:
    warnings.warn("Versions not found. Test mode.")
    _versions = ['test', current_version]
html_theme_options.update({
    'versions': [
        (c, f'../{c}/index.html')
        for c in set(_versions).union([current_version])
    ]
})

# Get logo
html_logo = "images/tda_logo.svg"
html_favicon = 'images/tda_favicon.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['source/_static/style.css']  # []  # ['_static']

html_sourcelink_suffix = ''

rst_epilog = """
.. |ProjectVersion| replace:: Foo Project, version {versionnum}
""".format(
    versionnum=release,
)

supported_image_types = [
    'image/svg+xml',
    'image/gif',
    'image/jpeg'
]
