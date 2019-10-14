"""The module :mod:`giotto.diagrams` implements persistence diagrams
transformers.
It offers the possibility to stack across homology dimensions,
scale, and filter diagrams.
It also makes it possible to calculate distance matrices.
"""

from .distance import PairwiseDistance, Amplitude
from .features import PersistenceEntropy, BettiCurve, PersistenceLandscape, \
    HeatKernel
from .preprocessing import ForgetDimension, Scaler, Filtering

__all__ = [
    'ForgetDimension',
    'BettiCurve',
    'PersistenceLandscape',
    'HeatKernel',
    'PersistenceEntropy',
    'Amplitude',
    'Scaler',
    'Filtering',
    'PairwiseDistance',
]
