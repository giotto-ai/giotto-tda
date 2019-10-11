.. giotto documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to giotto's API reference!
==================================


:mod:`giotto.time_series`: Preprocessing for time series
========================================================

.. automodule:: giotto.time_series
   :no-members:
   :no-inherited-members:

.. currentmodule:: giotto

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series.Resampler
   time_series.Stationarizer

   time_series.TakensEmbedding

   time_series.OrdinalRepresentation
   time_series.Entropy


:mod:`giotto.graphs`: Preprocessing for graphs
==============================================

.. automodule:: giotto.graphs
   :no-members:
   :no-inherited-members:

.. currentmodule:: giotto

.. autosummary::
   :toctree: generated/
   :template: class.rst

   graphs.TransitionGraph
   graphs.KNeighborsGraph
   graphs.GraphGeodesicDistance


..
   :mod:`giotto.images`: Preprocessing for images
   ==============================================

   .. automodule:: giotto.images
      :no-members:
      :no-inherited-members:

   .. currentmodule:: giotto

   .. autosummary::
      :toctree: generated/
      :template: class.rst

       images.ImageInverter
       images.HeightFiltration
       images.RadialFiltration
       images.DilationFiltration
       images.ErosionFiltration
       images.SignedDistanceFiltration
       images.DensityFiltration


:mod:`giotto.base`: Base
========================

.. automodule:: giotto.base
   :no-members:
   :no-inherited-members:

.. currentmodule:: giotto

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.TransformerResamplerMixin


:mod:`giotto.pipeline`: Pipeline
================================

.. automodule:: giotto.pipeline
   :no-members:
   :no-inherited-members:

.. currentmodule:: giotto

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pipeline.Pipeline

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pipeline.make_pipeline


:mod:`giotto.homology`: Persistent homology
===========================================

.. automodule:: giotto.homology
   :no-members:
   :no-inherited-members:

.. currentmodule:: giotto

.. autosummary::
   :toctree: generated/
   :template: class.rst

   homology.VietorisRipsPersistence
   ..
      homology.CubicalPersistence

   homology.ConsistentRescaling


:mod:`giotto.diagram`: Persistence diagrams
==========================================

.. automodule:: giotto.diagram
   :no-members:
   :no-inherited-members:

.. currentmodule:: giotto

.. autosummary::
   :toctree: generated/
   :template: class.rst

   diagram.Stacking
   diagram.Scaler
   diagram.Filtering

   diagram.DiagramDistance
   diagram.DiagramAmplitude

   diagram.PersistenceEntropy
   diagram.PersistenceLandscape
   diagram.BettiCurve
   diagram.HeatKernel

..
   :mod:`giotto.manifold`: Manifold learning
   =========================================

   .. automodule:: giotto.manifold
      :no-members:
      :no-inherited-members:

   .. currentmodule:: giotto

   .. autosummary::
      :toctree: generated/
      :template: class.rst

      manifold.StatefulMDS

      manifold.Kinematics


..
   :mod:`giotto.compose`: Features and targets composition
   =======================================================

   .. automodule:: giotto.compose
      :no-members:
      :no-inherited-members:

   .. currentmodule:: giotto

   .. autosummary::
      :toctree: generated/
      :template: class.rst

      compose.FeatureAggregator


..
   :mod:`giotto.neural_network`: Neural network with Keras
   =======================================================

   .. automodule:: giotto.neural_network
      :no-members:
      :no-inherited-members:

   .. currentmodule:: giotto

   .. autosummary::
      :toctree: generated/
      :template: class.rst

      neural_network.KerasClassifierWrapper
      neural_network.KerasRegressorWrapper

..
   :mod:`giotto.model_selection`: Hyper-parameter search extension
   ===============================================================

   .. automodule:: giotto.model_selection
      :no-members:
      :no-inherited-members:

   .. currentmodule:: giotto

   .. autosummary::
      :toctree: generated/
      :template: class.rst

      model_selection.GridSearchCV
      model_selection.RandomizedSearchCV
