
========================
Giotto-tda documentation
========================

**A high performance topological machine learning toolbox in Python.**

Giotto-tda is a high performance topological machine learning toolbox in Python built on top of
scikit-learn and is distributed under the GNU AGPLv3 license. It is part of the `Giotto <https://github.com/giotto-ai>`_ family of open-source projects.

**Read our paper on arXiv!**

Guiding principles
==================

 * | Seamless integration with widely used ML frameworks: Python + scikit-learn.
   | Inherit their strengths and allow for creation of heterogeneous ML pipelines.
 * | Code modularity: Algorithms as transformers
   | “Lego blocks” approach.
 * | User-friendliness and familiarity to the broad data science community.
   | Strict adherence to scikit-learn API and developer guidelines, “fit-transform” paradigm
 * | Standardisation.
   | Allow for integration of most available techniques into a generic framework.  Consistency of API across different modules
 * | Performance within the language constraints.
   | Vectorized code, parallelism (likely in future: just-in-time compilation and more)
 * | Data structures.
   | Support for time series, graphs, images.

30s guide to giotto-tda
=======================

Introduction to giotto-tda
--------------------------

The `post <https://towardsdatascience.com/getting-started-with-giotto-learn-a-python-library-for-topological-machine-learning-451d88d2c4bc>`_
by Lewis Tunstall provides a general overview of the library.

Mapper example
--------------

.. include::
   mapper_quickstart

Installation
============

.. include:: ../README.rst
   :start-after: Installation
   :end-before: Contributing

Developer installation
----------------------

For information about the developer installation, please refer to :ref:`Contributing guidelines <contrib>`.

API reference
=============
See :doc:`the API reference <api>`.

Examples
========
..
   Please have a look at :doc:`gallery/index`.

.. include:: notebooks/notebooks_readme.rst
   :start-after: Examples

Theory
======

The theory behind Topological Data Analysis might be overwhelming.
For this reason, we provide a :doc:`theory/glossary`.

Release notes - 0.1.4
=====================

Library name change
-------------------
The library and GitHub repository have been renamed to ``giotto-tda``! While the
new name is meant to better convey the library's focus on Topology-powered
machine learning and Data Analysis, the commitment to seamless integration with
``scikit-learn`` will remain just as strong and a defining feature of the project.
Concurrently, the main module has been renamed from ``giotto`` to ``gtda`` in this
version. ``giotto-learn`` will remain on PyPI as a legacy package (stuck at v0.1.3)
until we have ensured that users and developers have fully migrated. The new PyPI
package ``giotto-tda`` will start at v0.1.4 for project continuity.

Short summary: install via ::

    pip install -U giotto-tda

and ``import gtda`` in your scripts or notebooks!

Change of license
-----------------

The license changes from Apache 2.0 to GNU AGPLv3 from this release on.

Major Features and Improvements
-------------------------------
-  Added a ``mapper`` submodule implementing the Mapper algorithm of Singh, Mémoli and Carlsson. The main tools are the
   functions ``make_mapper_pipeline``, ``plot_static_mapper_graph`` and ``plot_interactive_mapper_graph``. The first
   creates an object of class ``MapperPipeline`` which can be fit-transformed to data to create a Mapper graph in the
   form of an ``igraph.Graph`` object (see below). The ``MapperPipeline`` class itself is a simple subclass
   of scikit-learn's ``Pipeline`` which is adapted to the precise structure of the Mapper algorithm, so that a
   ``MapperPipeline`` object can be used as part of even larger scikit-learn pipelines, inside a meta-estimator, in a
   grid search, etc. One also has access to other important features of scikit-learn's ``Pipeline``, such as memory
   caching to avoid unnecessary recomputation of early steps when parameters involved in later steps are changed.
   The clustering step can be parallelised over the pullback cover sets via ``joblib`` -- though this can actually
   *lower* performance in small- and medium-size datasets. A range of pre-defined filter functions are also included,
   as well as covers in one and several dimensions, agglomerative clustering algorithms based on stopping rules to
   create flat cuts, and utilities for making transformers out of callables or out of other classes which have no
   ``transform`` method. ``plot_static_mapper_graph`` allows the user to visualise (in 2D or 3D) the Mapper graph
   arising from fit-transforming a ``MapperPipeline`` to data, and offers a range of colouring options to correlate the
   graph's structure with exogenous or endogenous information. It relies on ``plotly`` for plotting and displaying
   metadata. ``plot_interactive_mapper_graph`` adds interactivity to this, via ``ipywidgets``: specifically, the user
   can fine-tune some parameters involved in the definition of the Mapper pipeline, and observe in real time how the
   structure of the graph changes as a result. In this release, all hyperparameters involved in the covering and
   clustering steps are supported. The ability to fine-tune other hyperparameters will be considered for future versions.
-  Added support for Python 3.8.

Bug Fixes
---------
-  Fixed consistently incorrect documentation for the ``fit_transform`` methods. This has been achieved by introducing a
   class decorator ``adapt_fit_transform_docs`` which is defined in the newly introduced ``gtda.utils._docs.py``.

Backwards-Incompatible Changes
------------------------------
-  The library name change and the change in the name of the main module ``giotto``
   are important major changes.
-  There are now additional dependencies in the ``python-igraph``, ``matplotlib``, ``plotly``, and ``ipywidgets`` libraries.

Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Umberto Lupo, Lewis Tunstall, Guillaume Tauzin, Philipp Weiler, Julian Burella Pérez.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

FAQ
===========

For additional information, please refer to :doc:`Frequently asked questions <faq>`.
