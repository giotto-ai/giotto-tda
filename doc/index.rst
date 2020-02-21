.. giotto-tda documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to giotto-tda's API reference!
======================================

Checkout the following resources:
 * :ref:`readme`,
 * :ref:`contrib`,
 * :ref:`theory`,
 * :doc:`gallery/index`

:mod:`gtda.mapper`: Mapper
==========================

.. include:: modules/mapper.rst

:mod:`gtda.homology`: Persistent homology
=========================================

.. automodule:: gtda.homology
   :no-members:
   :no-inherited-members:

.. include:: modules/homology.rst

:mod:`gtda.diagrams`: Persistence diagrams
==========================================

.. automodule:: gtda.diagrams
   :no-members:
   :no-inherited-members:

.. include:: modules/diagrams.rst

:mod:`gtda.time_series`: Time series
====================================

.. automodule:: gtda.time_series
   :no-members:
   :no-inherited-members:

Preprocessing
-------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/time_series/preprocessing/
   :template: class.rst

   time_series.SlidingWindow
   time_series.Resampler
   time_series.Stationarizer

Time-delay embedding
--------------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/time_series/embedding/
   :template: class.rst

   time_series.TakensEmbedding

Target preparation
------------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/time_series/target_preparation/
   :template: class.rst

   time_series.Labeller

Dynamical systems
-----------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/time_series/dynamical_system/
   :template: class.rst

   time_series.PermutationEntropy

Multivariate
------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/time_series/multivariate/
   :template: class.rst

   time_series.PearsonDissimilarity


:mod:`gtda.graphs`: Graphs
==========================

.. automodule:: gtda.graphs
   :no-members:
   :no-inherited-members:

Graph creation
--------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/graphs/creation/
   :template: class.rst

   graphs.TransitionGraph
   graphs.KNeighborsGraph

Graph processing
----------------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/graphs/processing/
   :template: class.rst

   graphs.GraphGeodesicDistance


:mod:`gtda.base`: Base
======================

.. automodule:: gtda.base
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.TransformerResamplerMixin


:mod:`gtda.pipeline`: Pipeline
==============================

.. automodule:: gtda.pipeline
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pipeline.Pipeline

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pipeline.make_pipeline


:mod:`gtda.meta_transformers`: Convenience pipelines
====================================================

.. automodule:: gtda.meta_transformers
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/
   :template: class.rst

   meta_transformers.EntropyGenerator
   meta_transformers.BettiCurveGenerator
   meta_transformers.LandscapeGenerator


:mod:`gtda.utils`: Validation
=============================

.. automodule:: gtda.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_diagram
   utils.validate_params
   utils.validate_metric_params

..
   :mod:`gtda.images`: Images
   ==========================

..
   automodule:: gtda.images
   :no-members:
   :no-inherited-members:

..
   currentmodule:: gtda/images/

..
   autosummary::
   :toctree: generated/
   :template: class.rst

   images.ImageInverter
   images.HeightFiltration
   images.RadialFiltration
   images.DilationFiltration
   images.ErosionFiltration
   images.SignedDistanceFiltration
   images.DensityFiltration


..
   :mod:`gtda.manifold`: Manifold learning
   =======================================

..
   automodule:: gtda.manifold
   :no-members:
   :no-inherited-members:

..
   currentmodule:: gtda

..
   autosummary::
   :toctree: generated/
   :template: class.rst

   manifold.StatefulMDS

   manifold.Kinematics
