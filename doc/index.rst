.. giotto-learn documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to giotto-learn's API reference!
========================================


:mod:`giottolearn.mapper`: Mapper
=================================

.. automodule:: giottolearn.mapper
   :no-members:
   :no-inherited-members:

Filters
-------
.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapper.Projection
   mapper.Eccentricity
   mapper.Entropy

Covers
-------
.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapper.OneDimensionalCover
   mapper.CubicalCover

Clustering
----------
.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapper.FirstSimpleGap
   mapper.FirstHistogramGap

Pipeline
--------
.. currentmodule:: giottolearn

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
.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapper.visualization.plot_static_mapper_graph
   mapper.visualization.plot_interactive_mapper_graph

Utilities
---------
.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapper.utils.decorators.method_to_transform
   mapper.utils.pipeline.transformer_from_callable_on_rows


:mod:`giottolearn.homology`: Persistent homology
================================================

.. automodule:: giottolearn.homology
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   homology.VietorisRipsPersistence
   homology.ConsistentRescaling


:mod:`giottolearn.diagrams`: Persistence diagrams
=================================================

.. automodule:: giottolearn.diagrams
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   diagrams.ForgetDimension
   diagrams.Scaler
   diagrams.Filtering

   diagrams.PairwiseDistance
   diagrams.Amplitude

   diagrams.PersistenceEntropy
   diagrams.PersistenceLandscape
   diagrams.BettiCurve
   diagrams.HeatKernel


:mod:`giottolearn.time_series`: Time series
===========================================

.. automodule:: giottolearn.time_series
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series.Resampler
   time_series.Stationarizer
   time_series.TakensEmbedding
   time_series.SlidingWindow
   time_series.PermutationEntropy
   time_series.PearsonDissimilarity
   time_series.Labeller


:mod:`giottolearn.graphs`: Graphs
=================================

.. automodule:: giottolearn.graphs
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   graphs.TransitionGraph
   graphs.KNeighborsGraph
   graphs.GraphGeodesicDistance
   

:mod:`giottolearn.base`: Base
=============================

.. automodule:: giottolearn.base
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.TransformerResamplerMixin


:mod:`giottolearn.pipeline`: Pipeline
=====================================

.. automodule:: giottolearn.pipeline
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pipeline.Pipeline

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pipeline.make_pipeline


:mod:`giottolearn.meta_transformers`: Convenience pipelines
===========================================================

.. automodule:: giottolearn.meta_transformers
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   meta_transformers.EntropyGenerator
   meta_transformers.BettiCurveGenerator
   meta_transformers.LandscapeGenerator


:mod:`giottolearn.utils`: Validation
====================================

.. automodule:: giottolearn.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: giottolearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_diagram
   utils.validate_params
   utils.validate_metric_params

..
   :mod:`giottolearn.images`: Images
   =================================

..
   automodule:: giottolearn.images
   :no-members:
   :no-inherited-members:

..
   currentmodule:: giottolearn

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
   :mod:`giottolearn.manifold`: Manifold learning
   ==============================================

..
   automodule:: giottolearn.manifold
   :no-members:
   :no-inherited-members:

..
   currentmodule:: giottolearn

..
   autosummary::
   :toctree: generated/
   :template: class.rst

   manifold.StatefulMDS

   manifold.Kinematics