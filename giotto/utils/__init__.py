"""The module :mod:`giotto.utils` implements hyperparameter and input
validation functions."""

from .validation import check_diagram, check_graph
from .validation import validate_metric_params, validate_params

import warnings

warnings.warn(
    "Starting at v0.1.4, this package was renamed as 'giotto-tda'. The "
    "giotto-learn PyPI package will no longer be developed or maintained, and "
    "will remain at the state of v0.1.3. Please visit "
    "https://github.com/giotto-ai/giotto-tda to find installation information "
    "for giotto-tda.")

__all__ = [
    'check_diagram',
    'check_graph',
    'validate_metric_params',
    'validate_params'
]
