"""The module :mod:`giotto.meta_transformers` implements convenience
:class:`giotto.pipeline.Pipeline` transformers for direct topological feature
generation."""

from .features import EntropyGenerator, BettiCurveGenerator, LandscapeGenerator

import warnings

warnings.warn(
    "Starting at v0.1.4, this package was renamed as 'giotto-tda'. The "
    "giotto-learn PyPI package will no longer be developed or maintained, and "
    "will remain at the state of v0.1.3. Please visit "
    "https://github.com/giotto-ai/giotto-tda to find installation information "
    "for giotto-tda.")

__all__ = [
    'EntropyGenerator',
    'BettiCurveGenerator',
    'LandscapeGenerator'
]
