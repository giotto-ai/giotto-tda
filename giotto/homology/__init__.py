"""The module :mod:`giotto.homology` implements transformers
to modify metric spaces or generate persistence diagrams.
"""

from .consistent import ConsistentRescaling
from .point_clouds import VietorisRipsPersistence

import warnings

warnings.warn(
    "Starting at v0.1.4, this package was renamed as 'giotto-tda'. The "
    "giotto-learn PyPI package will no longer be developed or maintained, and "
    "will remain at the state of v0.1.3. Please visit "
    "https://github.com/giotto-ai/giotto-tda to find installation information "
    "for giotto-tda.")


__all__ = [
    'VietorisRipsPersistence',
    'ConsistentRescaling',
]
