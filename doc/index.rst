.. topological_learning documentation master file, created by
   sphinx-quickstart on Mon Jun  3 11:56:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Topological Learning's API reference!
================================================


:mod:`topological_learning.preprocessing`: Preprocessing for time series
========================================================================

.. automodule:: topological_learning.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: topological_learning

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.Resampler
   preprocessing.Stationarizer

   preprocessing.TakensEmbedder


:mod:`topological_learning.homology`: Persistent homology
=========================================================

.. automodule:: topological_learning.homology
   :no-members:
   :no-inherited-members:

.. currentmodule:: topological_learning

.. autosummary::
   :toctree: generated/
   :template: class.rst

   homology.VietorisRipsPersistence
   homology.PersistentEntropy

   homology.ConsistentRescaling


:mod:`topological_learning.diagram`: Persistent diagrams
========================================================

.. automodule:: topological_learning.diagram
   :no-members:
   :no-inherited-members:

.. currentmodule:: topological_learning

.. autosummary::
   :toctree: generated/
   :template: class.rst

   diagram.DiagramStacker
   diagram.DiagramScaler
   diagram.DiagramFilter

   diagram.DiagramDistance

:mod:`topological_learning.manifold`: Manifold learning
=======================================================

.. automodule:: topological_learning.manifold
   :no-members:
   :no-inherited-members:

.. currentmodule:: topological_learning

.. autosummary::
   :toctree: generated/
   :template: class.rst

   manifold.StatefulMDS

   manifold.Kinematics


:mod:`topological_learning.compose`: Features and targets composition
=====================================================================

.. automodule:: topological_learning.compose
   :no-members:
   :no-inherited-members:

.. currentmodule:: topological_learning

.. autosummary::
   :toctree: generated/
   :template: class.rst

   compose.TargetResampler
   compose.FeatureAggregator

   compose.TargetResamplingClassifier
   compose.TargetResamplingRegressor


:mod:`topological_learning.neural_network`: Neural network with Keras
=====================================================================

.. automodule:: topological_learning.neural_network
   :no-members:
   :no-inherited-members:

.. currentmodule:: topological_learning

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neural_network.KerasClassifierWrapper
   neural_network.KerasRegressorWrapper


:mod:`topological_learning.model_selection`: Hyper-parameter search extension
=============================================================================

.. automodule:: topological_learning.model_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: topological_learning

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GridSearchCV
   model_selection.RandomizedSearchCV
