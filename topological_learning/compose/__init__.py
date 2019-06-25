"""The :mod:`topological_learning.compose` module is an extension of sk-learn's
and implements meta-estimators for building composite models with transformers
In addition to its current contents, this module will eventually be home to
refurbished versions of Pipeline and FeatureUnion.
"""

from .target import TargetResamplingClassifier, TargetResamplingRegressor, TargetResampler
from .features import FeatureAggregator

__all__ = [
    'TargetResamplingClassifier',
    'TargetResamplingRegressor',
    'TargetResampler',
    'FeatureAggregator'
]
