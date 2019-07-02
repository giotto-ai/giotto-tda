"""The :mod:`topological_learning.manifold` extends sk-learn's manifold learning
transformers.
"""

from .mds import StatefulMDS
from .kinematics import Kinematics


__all__ = [
    'StatefulMDS',
    'Kinematics'
]
