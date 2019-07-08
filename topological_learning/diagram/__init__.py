"""The :mod:`topological_learning.diagram` module implements persistence diagrams transformers.
It offers the possibility to stack across homology dimensions, scale, and filter diagrams.
It also makes it possible to calculate distance matrices.
"""

from .distance import DiagramDistance
from .preprocessing import DiagramStacker, DiagramScaler, DiagramFilter


__all__ = [
    'DiagramDistance',
    'DiagramStacker',
    'DiagramScaler',
    'DiagramFilter'
]
