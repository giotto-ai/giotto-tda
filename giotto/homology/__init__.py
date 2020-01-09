"""The module :mod:`giotto.homology` implements transformers
to modify metric spaces or generate persistence diagrams.
"""
# License: Apache 2.0

from .consistent import ConsistentRescaling
from .point_clouds import VietorisRipsPersistence
from .grids import CubicalPersistence

__all__ = [
    'VietorisRipsPersistence',
    'CubicalPersistence',
    'ConsistentRescaling',
]
