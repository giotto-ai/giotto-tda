.. giotto-learn documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to giotto-learn's API reference!
========================================


:mod:`glearn.mapper`: Mapper
============================

.. automodule:: glearn.mapper
   :no-members:
   :no-inherited-members:

Filters
-------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapper.Projection
   mapper.Eccentricity
   mapper.Entropy

Covers
-------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapper.OneDimensionalCover
   mapper.CubicalCover

Clustering
----------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapper.FirstSimpleGap
   mapper.FirstHistogramGap

Pipeline
--------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapper.pipeline.make_mapper_pipeline


.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapper.pipeline.MapperPipeline

Visualization
-------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapper.visualization.plot_static_mapper_graph
   mapper.visualization.plot_interactive_mapper_graph

Utilities
---------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapper.utils.decorators.method_to_transform
   mapper.utils.pipeline.transformer_from_callable_on_rows


:mod:`glearn.homology`: Persistent homology
===========================================

.. automodule:: glearn.homology
   :no-members:
   :no-inherited-members:

.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   homology.VietorisRipsPersistence
   homology.ConsistentRescaling


:mod:`glearn.diagrams`: Persistence diagrams
============================================

.. automodule:: glearn.diagrams
   :no-members:
   :no-inherited-members:

Preprocessing
-------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   diagrams.ForgetDimension
   diagrams.Scaler
   diagrams.Filtering

Distances
---------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   diagrams.PairwiseDistance

Diagram features
----------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   diagrams.Amplitude
   diagrams.PersistenceEntropy
   diagrams.PersistenceLandscape
   diagrams.BettiCurve
   diagrams.HeatKernel


:mod:`glearn.time_series`: Time series
======================================

.. automodule:: glearn.time_series
   :no-members:
   :no-inherited-members:

Preprocessing
-------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series.SlidingWindow
   time_series.Resampler
   time_series.Stationarizer

Time-delay embedding
--------------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series.TakensEmbedding

Target preparation
------------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series.Labeller

Dynamical systems
-----------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series.PermutationEntropy

Multivariate
------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series.PearsonDissimilarity


:mod:`glearn.graphs`: Graphs
============================

.. automodule:: glearn.graphs
   :no-members:
   :no-inherited-members:

Graph creation
--------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   graphs.TransitionGraph
   graphs.KNeighborsGraph

Graph processing
----------------
.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   graphs.GraphGeodesicDistance
   

:mod:`glearn.base`: Base
========================

.. automodule:: glearn.base
   :no-members:
   :no-inherited-members:

.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.TransformerResamplerMixin


:mod:`glearn.pipeline`: Pipeline
================================

.. automodule:: glearn.pipeline
   :no-members:
   :no-inherited-members:

.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pipeline.Pipeline

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pipeline.make_pipeline


:mod:`glearn.meta_transformers`: Convenience pipelines
======================================================

.. automodule:: glearn.meta_transformers
   :no-members:
   :no-inherited-members:

.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   meta_transformers.EntropyGenerator
   meta_transformers.BettiCurveGenerator
   meta_transformers.LandscapeGenerator


:mod:`glearn.utils`: Validation
===============================

.. automodule:: glearn.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: glearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_diagram
   utils.validate_params
   utils.validate_metric_params

..
   :mod:`glearn.images`: Images
   ============================

..
   automodule:: glearn.images
   :no-members:
   :no-inherited-members:

..
   currentmodule:: glearn

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
   :mod:`glearn.manifold`: Manifold learning
   =========================================

..
   automodule:: glearn.manifold
   :no-members:
   :no-inherited-members:

..
   currentmodule:: glearn

..
   autosummary::
   :toctree: generated/
   :template: class.rst

   manifold.StatefulMDS

   manifold.Kinematics