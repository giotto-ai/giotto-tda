"""The module :mod:`giotto.time_series` implements preprocessing techniques
   useful to apply Topological Data Analysis techniques to time series.
"""

from .embedding import SlidingWindow, TakensEmbedding
from .features import PermutationEntropy
from .preprocessing import Resampler, Stationarizer
from .multivariate import PearsonCorrelation
from .target import Labeller

__all__ = [
    'Resampler',
    'Stationarizer',
    'PermutationEntropy',
    'TakensEmbedding',
    'SlidingWindow',
    'Labeller',
    'PearsonCorrelation'
]
