"""The :mod:`topological_learning.model_selection` extends sk-learn's
model_selection techniques to extends their compatibility with Keras-based
Pipelines.
"""

from ._search import GridSearchCV, RandomizedSearchCV


__all__ = [
    'GridSearchCV'
    'RandomizedSearchCV'
]
