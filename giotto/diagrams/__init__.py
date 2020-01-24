"""The module :mod:`giotto.diagrams` implements transformers to
preprocess persistence diagrams or extract features from them.
"""

from .preprocessing import ForgetDimension, Scaler, Filtering
from .distance import PairwiseDistance, Amplitude
from .features import PersistenceEntropy, BettiCurve, PersistenceLandscape, \
    HeatKernel

import warnings

warnings.warn(
    "Starting at v0.1.4, this package was renamed as 'giotto-tda'. The "
    "giotto-learn PyPI package will no longer be developed or maintained, and "
    "will remain at the state of v0.1.3. Please visit "
    "https://github.com/giotto-ai/giotto-tda to find installation information "
    "for giotto-tda.")

__all__ = [
    'ForgetDimension',
    'Scaler',
    'Filtering',
    'PairwiseDistance',
    'Amplitude',
    'BettiCurve',
    'PersistenceLandscape',
    'HeatKernel',
    'PersistenceEntropy'
]
