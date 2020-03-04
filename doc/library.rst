
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

For installation instructions, see :ref:`the installation instructions <installation>`.

The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers.
For more complex architectures, you should use the Keras functional API, which allows to build arbitrary graphs of layers.

The functionalities of Giotto-TDA are provided in `scikit-learn`-style transformers.
This allows you to jump in the familiar way of generating features from your data. Here is an example of `VietorisRipsPersistence`,

.. code-block:: python

  from gtda.homology import VietorisRipsPersistence
  vietorisrips_tr = VietorisRipsPersistence()

which calculates :ref:` persistent diagrams <persistence diagram>` from a :ref:`point cloud <finite metric spaces and point clouds>`.

.. code-block:: python

  diagrams = vietorisrips_tr.fit_transform(point_clouds)

You can create features from the persistence diagrams in a similar way, using, for example, the :ref:`BettiCurve <BettiCurve>` transformer.

.. code-block:: python

  from gtda.diagrams import PersistenceEntropy
  image_tr = PersistenceEntropy()
  features = image_tr.fit_transform(diagrams)

Extraction of topological features phrased in terms of transformers allows you to use the standard tools from scikit-learn,
combining topological features with standard models and
like grid-search or cross-validate your model.

.. code-block:: python

  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split

  X_train, X_valid, y_train, y_valid = train_test_split(features, labels)
  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  model.score(X_valid, y_valid)

Examples
========

We provide a number of :ref:`examples <notebooks_index>`, which offer:

 - a more comprehensive view of how the API can be used in simple, synthetic examples,
 - an intuitive explanation of topological techniques.

Additionally, the `post <https://towardsdatascience.com/getting-started-with-giotto-learn-a-python-library-for-topological-machine-learning-451d88d2c4bc>`_
by Lewis Tunstall provides a general overview of the library.

Use cases
=========

A selection of use cases that we worked on are available as github repositories and some of them are published as posts on Medium.
Please refer to `github <https://github.com/giotto-ai>` or the `L2F website <https://giotto.ai/learn/course-content>` for a summary.

Mapper notebook
======================
The mapper notebook, :doc:`included in the documentation <notebooks/mapper_quickstart.rst>`,
provides a comprehensive introduction to mapper.

**********
What's new
**********

.. include::
   release.rst
   :start-after: Release 0.1.4
   :end-before: Release 0.1.3