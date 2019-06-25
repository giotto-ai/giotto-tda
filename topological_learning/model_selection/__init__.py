"""The :mod:`topological_learning.model_selection` extends sk-learn's
model_selection techniques.
"""

from ._search import GridSearchCV


__all__ = [
    'GridSearchCV'
]
