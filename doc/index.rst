.. giotto documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
    :maxdepth: 2

    glossary
    modules/compose.rst
    modules/diagrams.rst
    modules/homology.rst
    modules/manifold.rst
    modules/model_selection.rst
    modules/time_series.rst

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
   time_series.SlidingWindow
   time_series.TakensEmbedding


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

..
   automodule:: giotto.images
   :no-members:
   :no-inherited-members:

..
   currentmodule:: giotto

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
   homology.ConsistentRescaling


:mod:`giotto.diagrams`: Persistence diagrams
============================================

.. automodule:: giotto.diagrams
   :no-members:
   :no-inherited-members:

.. currentmodule:: giotto

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

:mod:`giotto.utils`: Validation
===============================

.. automodule:: giotto.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: giotto

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_diagram
   utils.validate_params
   utils.validate_metric_params

..
   :mod:`giotto.manifold`: Manifold learning
   =========================================

..
   automodule:: giotto.manifold
   :no-members:
   :no-inherited-members:

..
   currentmodule:: giotto

..
   autosummary::
   :toctree: generated/
   :template: class.rst

   manifold.StatefulMDS

   manifold.Kinematics


..
   :mod:`giotto.compose`: Features and targets composition
   =======================================================

..
   automodule:: giotto.compose
   :no-members:
   :no-inherited-members:

..
   currentmodule:: giotto

..
   autosummary::
   :toctree: generated/
   :template: class.rst

   compose.FeatureAggregator


..
   :mod:`giotto.neural_network`: Neural network with Keras
   =======================================================

..
   automodule:: giotto.neural_network
   :no-members:
   :no-inherited-members:
..
   currentmodule:: giotto

..
   autosummary::
   :toctree: generated/
   :template: class.rst

   neural_network.KerasClassifierWrapper
   neural_network.KerasRegressorWrapper

..
   :mod:`giotto.model_selection`: Hyper-parameter search extension
   ===============================================================

..
   automodule:: giotto.model_selection
   :no-members:
   :no-inherited-members:

..
   currentmodule:: giotto

..
   autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GridSearchCV
   model_selection.RandomizedSearchCV
