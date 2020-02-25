
========================
Giotto-tda documentation
========================

**A high performance topological machine learning toolbox in Python.**

Giotto-tda is a high performance topological machine learning toolbox in Python built on top of
scikit-learn and is distributed under the GNU AGPLv3 license. It is part of the `Giotto <https://github.com/giotto-ai>`_ family of open-source projects.


Checkout the following resources:
 * :ref:`readme`,
 * :ref:`contrib`,
 * :ref:`theory`,
 * :doc:`gallery/index`

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

**Read our paper on arXiv!**

30s guide to giotto-tda
=======================

Include the mapper quickstart notebook and Lewi's Medium post.


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
Please have a look at :doc:`gallery/index`.
