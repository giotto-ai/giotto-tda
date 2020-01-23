"""The module :mod:`gtda.homology` implements transformers
to modify metric spaces or generate persistence diagrams.
"""
# License: GNU AGPLv3

from .consistent import ConsistentRescaling
from .point_clouds import VietorisRipsPersistence


__all__ = [
    'VietorisRipsPersistence',
    'ConsistentRescaling'
]
