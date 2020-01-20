Release 0.1.4
=============

Change of license
-----------------

The license changes from Apache 2.0 to AGPLv3 from this release on.

Major Features and Improvements
-------------------------------
-  Added `mapper` submodule implementing the Mapper algorithm with all its steps, some clustering algorithms, joblib
   parallelism at the level of clustering, static/interactive visualization tools.
-  Added support for Python 3.8.
-  Renaming of the main `giotto` module as `glearn` (see below).

Bug Fixes
---------
-  Fixed consistently incorrect documentation for the `fit_transform` methods. This has been achieved by introducing a
   class decorator `adapt_fit_transform_docs` which is defined in the newly introduced `glearn.utils._docs.py`.

Backwards-Incompatible Changes
------------------------------
-  The main module `giotto` has been renamed to `glearn` for 1) similarity with scikit-learn -> `sklearn`, making it
   easy to memorize, 2) brevity re e.g. `giottolearn`, and 3) consistency with the existence of other libraries in the
   Giotto ecosystem.
-  There are now additional dependencies in the `python-igraph`, `matplotlib`, `plotly`, and `ipywidgets` libraries.

Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Umberto Lupo, Lewis Tunstall, Guillaume Tauzin, Philipp Weiler, Julian Burella Pérez.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


Release 0.1.3
=============

Major Features and Improvements
-------------------------------
None

Bug Fixes
---------
-  Fixed a bug in `diagrams.Amplitude` causing the transformed array to be wrongly filled and added adequate test.

Backwards-Incompatible Changes
------------------------------
None.

Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Umberto Lupo.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


Release 0.1.2
=============

Major Features and Improvements
-------------------------------
-  Added support for Python 3.5.

Bug Fixes
---------
None.

Backwards-Incompatible Changes
------------------------------
None.

Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Matteo Caorsi, Henry Tom (@henrytomsf), Guillaume Tauzin.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


Release 0.1.1
=============

Major Features and Improvements
-------------------------------
-  Improved documentation.
-  Improved features of class `Labeller`.
-  Improved features of class `PearsonDissimilarities`.
-  Improved GitHub files.
-  Improved CI.

Bug Fixes
---------
Fixed minor bugs from the first release.

Backwards-Incompatible Changes
------------------------------
The following class were renamed:
-  class `PearsonCorrelation` was renamed to class`PearsonDissimilarities`

Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Umberto Lupo, Guillaume Tauzin, Matteo Caorsi, Olivier Morel.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


Release 0.1.0
=============

Major Features and Improvements
-------------------------------

The following submodules where added:

-  `giotto.homology` implements transformers to modify metric spaces or generate persistence diagrams.
-  `giotto.diagrams` implements transformers to preprocess persistence diagrams or extract features from them.
-  `giotto.time_series` implements transformers to preprocess time series or embed them in a higher dimensional space for persistent homology.
-  `giotto.graphs` implements transformers to create graphs or extract metric spaces from graphs.
-  `giotto.meta_transformers` implements convenience `giotto.Pipeline` transformers for direct topological feature generation.
-  `giotto.utils` implements hyperparameters and input validation functions.
-  `giotto.base` implements a TransformerResamplerMixin for transformers that have a resample method.
-  `giotto.pipeline` extends scikit-learn's module by defining Pipelines that include TransformerResamplers.


Bug Fixes
---------


Backwards-Incompatible Changes
------------------------------


Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Guillaume Tauzin, Umberto Lupo, Philippe Nguyen, Matteo Caorsi, Julian Burella Pérez,
Alessio Ghiraldello.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


Release 0.1a.0
==============

Initial release of giotto-learn.
