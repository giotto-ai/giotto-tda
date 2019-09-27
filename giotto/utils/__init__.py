"""
The :mod:`giotto.utils` module includes various utilities.
"""

from .validation import check_diagram
from .validation import validate_metric_params, validate_params


__all__ = [
    'check_diagram',
    'validate_metric_params',
    'validate_params'
]
