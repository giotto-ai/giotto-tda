"""The module :mod:`giotto.time_series` implements transformers to
preprocess time series or embed them in a higher dimensional space for
persistent homology.
"""

from .embedding import SlidingWindow, TakensEmbedding
from .features import PermutationEntropy
from .preprocessing import Resampler, Stationarizer
from .multivariate import PearsonDissimilarity
from .target import Labeller

import warnings

warnings.warn(
    "Starting at v0.1.4, this package was renamed as 'giotto-tda'. The "
    "giotto-learn PyPI package will no longer be developed or maintained, and "
    "will remain at the state of v0.1.3. Please visit "
    "https://github.com/giotto-ai/giotto-tda to find installation information "
    "for giotto-tda.")

__all__ = [
    'Resampler',
    'Stationarizer',
    'PermutationEntropy',
    'TakensEmbedding',
    'SlidingWindow',
    'Labeller',
    'PearsonDissimilarity'
]
