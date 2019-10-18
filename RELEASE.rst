Release 0.1.0
=============

Major Features and Improvements
-------------------------------

The following submodules where added:

-  :mod:`giotto.homology` implements transformers to generate persistence diagrams.
-  :mod:`giotto.diagrams` implements transformers to preprocess diagrams and extract features from them.
-  :mod:`giotto.time_series` implements transformers to preprocess time series for persitent homology.
-  :mod:`giotto.graphs` implements preprocessing techniques useful to apply persistent homology to graphs.
-  :mod:`giotto.meta_transformers` implements meta transformers for direct topological feature generation..
-  :mod:`giotto.base` implements TransformerResamplerMixin for transformer that have a resample method.
-  :mod:`giotto.pipeline` extends sklearn's module for Pipelines that include TransformerResamplers.
-  :mod:`giotto.utils` implements hyperparameters and input validation functions.


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
