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
import sphinx_rtd_theme

from gtda import __version__

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'giotto-tda'
copyright = '2020, L2F'
author = 'Guillaume Tauzin, Umberto Lupo, Matteo Caorsi, Anibal Medina, ' \
         'Lewis Tunstall'

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
    'sphinx.ext.imgconverter',
    #'sphinx_gallery.gen_gallery',
    #'sphinx_nbexamples',
    #'ipypublish.sphinx.notebook',
    'nbsphinx',
    'sphinx_issues',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon'
    # 'custom_references_resolver' # custom for sklearn, not sure what it does
]

# Add mappings
intersphinx_mapping = {
    'sklearn': ('http://scikit-learn.org/stable', None),
    'plotly': ('https://plot.ly/python-api-reference/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None)
}

sphinx_gallery_conf = {
    'examples_dirs': '../examples/',   # path to your example scripts
    'gallery_dirs': 'gallery',  # path to where to save gallery generated output
}

# ipypublish.sphinx.notebook
source_suffix = {
    '.ipynb': 'jupyter_notebook',
}
numfig = True
math_numfig = True
numfig_secnum_depth = 2

math_number_all = True


# Sphinx-nbexamples
process_examples = True
example_gallery_config = dict(
    examples_dirs='../examples/',
    gallery_dirs='gallery_nb',
    pattern='*.ipynb',

)


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
templates_path = ['templates']

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
#
#html_theme = 'nature'
html_theme = "sphinx_rtd_theme"

path_to_image = 'images/tda_logo.svg'
if os.path.exists(path_to_image):
    import requests
    r = requests.get('https://www.giotto.ai/static/vector/logo-tda.svg')
    with open(path_to_image, 'w') as f:
        f.write(r.content)
html_logo = path_to_image

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': False,
    'logo_only': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['source/_static/style.css'] # []  # ['_static']


#scv_whitelist_branches = ('master', 'w_persistent_image', 'w_cubical', 'w-p-igraph-dep',
#                          'time_series_tests', 'test_features', 'silhouette')
#scv_whitelist_tags = ('v0.1a.0', 'v0.1.0', 'v0.1.1', 'v0.1.2')
#scv_whitelist_tags = ('v0.1.4', 'v0.1.3')
#scv_whitelist_branches = ('ghpages', )

rst_epilog = """
.. |ProjectVersion| replace:: Foo Project, version {versionnum}
""".format(
    versionnum=release,
)