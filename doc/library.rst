########
Overview
########

**A high performance topological machine learning toolbox in Python.**

Giotto-tda is a high performance topological machine learning toolbox in Python built on top of
scikit-learn and is distributed under the GNU AGPLv3 license. It is part of the `Giotto <https://github.com/giotto-ai>`_ family of open-source projects.

**Read our paper on arXiv!**

******************
Guiding principles
******************

 * | **Seamless integration with widely used ML frameworks: Python + scikit-learn.**
   | Inherit their strengths and allow for creation of heterogeneous ML pipelines.
 * | **Code modularity: Algorithms as transformers.**
   | “Lego blocks” approach.
 * | **User-friendliness and familiarity to the broad data science community.**
   | Strict adherence to scikit-learn API and developer guidelines, “fit-transform” paradigm
 * | **Standardisation.**
   | Allow for integration of most available techniques into a generic framework.  Consistency of API across different modules
 * | **Performance within the language constraints.**
   | Vectorized code, parallelism (likely in future: just-in-time compilation and more)
 * | **Data structures.**
   | Support for time series, graphs, images.

***********************
30s guide to Giotto-TDA
***********************

For installation instructions, see :ref:`the developer installation instructions <installation>`.

The `post <https://towardsdatascience.com/getting-started-with-giotto-learn-a-python-library-for-topological-machine-learning-451d88d2c4bc>`_
by Lewis Tunstall provides a general overview of the library.

The mapper notebook, :doc:`included in the documentation <notebooks/mapper_quickstart.rst>`,
provides a comprehensive introduction to mapper.

**********
What's new
**********

.. include::
   release.rst
   :start-after: Release 0.1.4
   :end-before: Release 0.1.3