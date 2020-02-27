"""The module :mod:`gtda.homology` implements transformers
to process point clouds and modify metric spaces.
"""
# License: GNU AGPLv3

from .rescaling import ConsistentRescaling, ConsecutiveRescaling

__all__ = [
    'ConsistentRescaling',
    'ConsecutiveRescaling',
]
