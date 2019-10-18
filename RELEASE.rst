Release 0.1.0
=============

Major Features and Improvements
-------------------------------

The following submodules where added:

-  :mod:`giotto.homology` implements transformers to modify metric spaces or generate persistence diagrams.
-  :mod:`giotto.diagrams` implements transformers to preprocess persistence diagrams or extract features from them.
-  :mod:`giotto.time_series` implements transformers to preprocess time series or embed them in a higher dimensional space for persistent homology.
-  :mod:`giotto.graphs` implements transformers to create graphs or extract metric spaces from graphs.
-  :mod:`giotto.meta_transformers` implements convenience :class:`giotto.Pipeline` transformers for direct topological feature generation.
-  :mod:`giotto.utils` implements hyperparameters and input validation functions.
-  :mod:`giotto.base` implements a TransformerResamplerMixin for transformers that have a resample method.
-  :mod:`giotto.pipeline` extends scikit-learn's module by defining Pipelines that include TransformerResamplers.


Bug Fixes
---------


Backwards-Incompatible Changes
------------------------------


Thanks to our Contributors
--------------------------

This release contains contributions from many people at Google, as well as:

Guillaume Tauzin, Umberto Lupo, Philippe Nguyen, Matteo Caorsi, Julian Burella Pérez,
Alessio Ghiraldello.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


Release 0.1a.0
==============

Initial release of giotto-learn.
