"""The module :mod:`giotto.diagram` implements persistence diagrams
transformers.
It offers the possibility to stack across homology dimensions,
scale, and filter diagrams.
It also makes it possible to calculate distance matrices.
"""

from .distance import DiagramDistance, DiagramAmplitude
from .preprocessing import DiagramStacker, DiagramScaler, DiagramFilter
from .features import PersistentEntropy

__all__ = [
    'DiagramStacker',
    'DiagramScaler',
    'DiagramFilter',
    'DiagramDistance',
    'DiagramAmplitude',
    'PersistentEntropy'
]
