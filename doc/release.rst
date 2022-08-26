#############
Release Notes
#############

.. _stable:

*************
Release 0.6.0
*************

This is a major release including a new local homology subpackage, a new backend for computing Vietoris–Rips barcodes, wheels for Python 3.10 and Apple Silicon systems, and end of support for Python 3.6.

Major Features and Improvements
===============================

- A new ``local_homology`` subpackage containing ``scikit-learn``–compatible transformers for the extraction of local homology features has been added (`#602 <https://github.com/giotto-ai/giotto-tda/pull/602>`_). A `tutorial <https://giotto-ai.github.io/gtda-docs/0.6.0/notebooks/local_homology.html>`_ and an `example <https://giotto-ai.github.io/gtda-docs/0.6.0/notebooks/local_hom_NLP_disambiguation.html>`_ notebooks explain it.
- Wheels for Python 3.10 are now available (`#644 <https://github.com/giotto-ai/giotto-tda/pull/644>`_ and `#646 <https://github.com/giotto-ai/giotto-tda/pull/646>`_).
- Wheels for Apple Silicon systems are now available for Python versions 3.8, 3.9 and 3.10 (`#646 <https://github.com/giotto-ai/giotto-tda/pull/646>`_).
- ``giotto-ph`` is now the backend for the computation of Vietoris–Rips barcodes, replacing ``ripser.py`` (`#614 <https://github.com/giotto-ai/giotto-tda/pull/614>`_).
- The documentation has been improved (`#609 <https://github.com/giotto-ai/giotto-tda/pull/609>`_).

Bug Fixes
=========

- A bug involving tests for the ``mapper`` subpackage has been fixed (`#638 <https://github.com/giotto-ai/giotto-tda/pull/638>`_).

Backwards-Incompatible Changes
==============================

- Python 3.6 is no longer supported, and the manylinux standard has been bumped from ``manylinux2010`` to ``manylinux2014`` (`#644 <https://github.com/giotto-ai/giotto-tda/pull/644>`_ and `#646 <https://github.com/giotto-ai/giotto-tda/pull/646>`_).
- The ``python-igraph`` requirement has been replaced with ``igraph >= 0.9.8`` (`#616 <https://github.com/giotto-ai/giotto-tda/pull/616>`_).

Thanks to our Contributors
==========================

This release contains contributions from:

Umberto Lupo, Jacob Bamberger, Wojciech Reise, Julián Burella Pérez, and Anibal Medina-Mardones

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

*************
Release 0.5.1
*************

This release was made shortly after the release of version 0.5.0, to resolve an important bug. Please refer to `the release notes for 0.5.0 <https://giotto-ai.github.io/gtda-docs/0.5.0/release.html#release-0-5-0>`_ to see the major improvements and backwards-incompatible changes to the Mapper subpackage which were introduced there.

Major Features and Improvements
===============================

None.

Bug Fixes
=========

A bug preventing Mapper pipelines from working with memory caching has been fixed (`#597 <https://github.com/giotto-ai/giotto-tda/pull/597>`_).

Backwards-Incompatible Changes
==============================

None.

Thanks to our Contributors
==========================

This release contains contributions from:

Umberto Lupo

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

*************
Release 0.5.0
*************

Major Features and Improvements
===============================

- An object-oriented API for interactive plotting of Mapper graphs has been added with the ``MapperInteractivePlotter`` (`#586 <https://github.com/giotto-ai/giotto-tda/pull/586>`_). This is intended to supersede ``plot_interactive_mapper`` graph as it allows for inspection of the current state of the objects change by interactivity. See also "Backwards-Incompatible Changes" below.
- Further citations have been added to the mathematical glossary (`#564 <https://github.com/giotto-ai/giotto-tda/pull/564>`_).

Bug Fixes
=========

- A bug preventing ``EuclideanCechPersistence`` from working correctly on point clouds in more than 2 dimensions has been fixed (`#588 <https://github.com/giotto-ai/giotto-tda/pull/588>`_).
- A validation bug preventing ``VietorisRipsPersistence`` and ``WeightedRipsPersistence`` from accepting non-empty dictionaries as ``metric_params`` has been fixed (`#590 <https://github.com/giotto-ai/giotto-tda/pull/590>`_).
- A bug causing an exception to be raised when ``node_color_statistic`` was passed as a numpy array in ``plot_static_mapper_graph`` has been fixed (`#576 <https://github.com/giotto-ai/giotto-tda/pull/576>`_).

Backwards-Incompatible Changes
==============================

- A major change to the behaviour of the (static and interactive) Mapper plotting functions ``plot_static_mapper_graph`` and ``plot_interactive_mapper_graph`` was introduced in `#584 <https://github.com/giotto-ai/giotto-tda/pull/584>`_. The new ``MapperInteractivePlotter`` class (see "Major Features and Improvements" above) also follows this new API. The main changes are as follows:

   - ``color_by_columns_dropdown``  has been eliminated.
   - ``color_variable`` has been renamed to ``color_features`` (but cannot be an array).
   - An additional keyword argument ``color_data`` has been added to more clearly separate the input ``data`` to the Mapper pipeline from the data to be used for coloring.
   - ``node_color_statistic`` is now applied column by column -- previously it could end up being applied to 2d arrays as a whole.
   - The defaults for color-related arguments lead to index values instead of the mean of the data.

- The default for ``weight_params`` in ``WeightedRipsPersistence`` is now the empty dictionary, and ``None`` is no longer allowed (`#595 <https://github.com/giotto-ai/giotto-tda/pull/595>`_).

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo, Wojciech Reise, Julian Burella Pérez, Sean Law, Anibal Medina-Mardones, and Lewis Tunstall

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

*************
Release 0.4.0
*************

Major Features and Improvements
===============================

- Wheels for Python 3.9 have been added (`#528 <https://github.com/giotto-ai/giotto-tda/pull/528>`_).
- Weighted Rips filtrations, and in particular distance-to-measure (DTM) based filtrations, are now supported in ``ripser`` and by the new ``WeightedRipsPersistence`` transformer (`#541 <https://github.com/giotto-ai/giotto-tda/pull/541>`_).
- See "Backwards-Incompatible Changes" for major improvements to ``ParallelClustering`` and therefore ``make_mapper_pipeline`` which are also major breaking changes.
- GUDHI's edge collapser can now be used with arbitrary vertex and edge weights (`#558 <https://github.com/giotto-ai/giotto-tda/pull/558>`_).
- ``GraphGeodesicDistance`` can now take rectangular input (the number of vertices is inferred to be ``max(x.shape)``), and ``KNeighborsGraph`` can now take sparse input (`#537 <https://github.com/giotto-ai/giotto-tda/pull/537>`_).
- ``VietorisRipsPersistence`` now takes a ``metric_params`` parameter (`#541 <https://github.com/giotto-ai/giotto-tda/pull/541>`_).

Bug Fixes
=========

- A documentation bug affecting plots from ``DensityFiltration`` has been fixed (`#540 <https://github.com/giotto-ai/giotto-tda/pull/540>`_).
- A bug affecting the bindings for GUDHI's edge collapser, which incorrectly did not ignore lower diagonal entries, has been fixed (`#538 <https://github.com/giotto-ai/giotto-tda/pull/538>`_).
- Symmetry conflicts in the case of sparse input to ``ripser`` and ``VietorisRipsPersistence`` are now handled in a way true to the documentation, i.e. by favouring upper diagonal entries if different values in transpose positions are also stored (`#537 <https://github.com/giotto-ai/giotto-tda/pull/537>`_).

Backwards-Incompatible Changes
==============================

- The minimum required version of ``pyflagser`` is now 0.4.3 (`#537 <https://github.com/giotto-ai/giotto-tda/pull/537>`_).
- ``ParallelClustering.fit_transform`` now outputs one array of cluster labels per sample, bringing it closer to ``scikit-learn`` convention for clusterers, and the fitted single clusterers are no longer stored in the ``clusterers_`` attribute of the fitted object (`#535 <https://github.com/giotto-ai/giotto-tda/pull/535>`_ and `#552 <https://github.com/giotto-ai/giotto-tda/pull/552>`_).

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo, Julian Burella Pérez, and Wojciech Reise.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

*************
Release 0.3.1
*************

Major Features and Improvements
===============================

- The latest changes made to the ``ripser.py`` submodule have been pulled (`#530 <https://github.com/giotto-ai/giotto-tda/pull/530>`_, see also `#532 <https://github.com/giotto-ai/giotto-tda/pull/532>`_). This includes in particular the performance improvements to the C++ backend submitted by Julian Burella Pérez via `scikit-tda/ripser.py#106 <https://github.com/scikit-tda/ripser.py/pull/106>`_. The developer installation now includes a new dependency in `robinhood hashmap <https://github.com/martinus/robin-hood-hashing>`_. These changes do not affect functionality.
- The example notebook `classifying_shapes.ipynb <https://github.com/giotto-ai/giotto-tda/blob/46b18a48205e5611f3c2e0eaa21072a93ada5bcb/examples/classifying_shapes.ipynb>`_ has been modified and improved (`#523 <https://github.com/giotto-ai/giotto-tda/pull/523>`_).
- The tutorial previously called ``time_series_classification.ipynb`` has been split into an introductory tutorial on the Takens embedding ideas (`topology_time_series.ipynb <https://github.com/wreise/giotto-tda/blob/b5321f5858eb12103a5f08126ad68d597b41aca9/examples/topology_time_series.ipynb>`_) and an example notebook on gravitational wave detection (`gravitational_waves_detection.ipynb <https://github.com/wreise/giotto-tda/blob/b5321f5858eb12103a5f08126ad68d597b41aca9/examples/gravitational_waves_detection.ipynb>`_) which presents a time series classification task (`#529 <https://github.com/giotto-ai/giotto-tda/pull/529>`_).
- The documentation for ``PairwiseDistance`` has been improved (`#525 <https://github.com/giotto-ai/giotto-tda/pull/525>`_).

Bug Fixes
=========

- Timeout deadlines for some of the ``hypothesis`` tests have been increased to make them less flaky (`#531 <https://github.com/giotto-ai/giotto-tda/pull/531>`_).

Backwards-Incompatible Changes
==============================

- Due to poor support for ``brew`` in the macOS 10.14 virtual machines by Azure, the CI for macOS systems is now run on 10.15 virtual machines and 10.14 is no longer supported by the wheels (`#527 <https://github.com/giotto-ai/giotto-tda/pull/527>`_)

Thanks to our Contributors
==========================

This release contains contributions from many people:

Julian Burella Pérez, Umberto Lupo, Lewis Tunstall, Wojciech Reise, and Rayna Andreeva.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

*************
Release 0.3.0
*************

Major Features and Improvements
===============================

This is a major release which adds substantial new functionality and introduces several improvements.

Persistent homology of directed flag complexes via ``pyflagser``
----------------------------------------------------------------

-  The ``pyflagser`` package (`source <https://github.com/giotto-ai/pyflagser>`_, `docs <https://docs-pyflagser.giotto.ai/>`_) is now an official dependency of ``giotto-tda``.
-  The ``FlagserPersistence`` transformer has been added to ``gtda.homology`` (`#339 <https://github.com/giotto-ai/giotto-tda/pull/339>`_). It wraps ``pyflagser.flagser_weighted`` to allow for computations of persistence diagrams from directed or undirected weighted graphs. A `new notebook <https://giotto-ai.github.io/gtda-docs/0.3.0/notebooks/persistent_homology_graphs.html>`_ demonstrates its use.

Edge collapsing and performance improvements for persistent homology
--------------------------------------------------------------------

-  GUDHI C++ components have been updated to the state of GUDHI v3.3.0, yielding performance improvements in ``SparseRipsPersistence``, ``EuclideanCechPersistence`` and ``CubicalPersistence`` (`#468 <https://github.com/giotto-ai/giotto-tda/pull/468>`_).
-  Bindings for GUDHI's `edge collapser <https://hal.inria.fr/hal-02395227>`_ have been created and can now be used as an optional preprocessing step via the optional keyword argument ``collapse_edges`` in ``VietorisRipsPersistence`` and in ``gtda.externals.ripser`` (`#469 <https://github.com/giotto-ai/giotto-tda/pull/469>`_ and `#483 <https://github.com/giotto-ai/giotto-tda/pull/483>`_). When ``collapse_edges=True``, and the input data and/or number of required homology dimensions is sufficiently large, the resulting runtimes for Vietoris–Rips persistent homology are state of the art.
-  The performance of the Ripser bindings has otherwise been improved by avoiding unnecessary data copies, better managing the memory, and using more efficient matrix routines (`#501 <https://github.com/giotto-ai/giotto-tda/pull/501>`_ and `#507 <https://github.com/giotto-ai/giotto-tda/pull/507>`_).

New transformers and functionality in ``gtda.homology``
-------------------------------------------------------

-  The ``WeakAlphaPersistence`` transformer has been added to ``gtda.homology`` (`#464 <https://github.com/giotto-ai/giotto-tda/pull/464>`_). Like ``VietorisRipsPersistence``, ``SparseRipsPersistence`` and ``EuclideanCechPersistence``, it computes persistent homology from point clouds, but its runtime can scale much better with size in low dimensions.
-  ``VietorisRipsPersistence`` now accepts sparse input when ``metric="precomputed"`` (`#424 <https://github.com/giotto-ai/giotto-tda/pull/424>`_).
-  ``CubicalPersistence`` now accepts lists of 2D arrays (`#503 <https://github.com/giotto-ai/giotto-tda/pull/503>`_).
-  A ``reduced_homology`` parameter has been added to all persistent homology transformers. When ``True``, one infinite bar in the H0 barcode is removed for the user automatically. Previously, it was not possible to *keep* these bars in the simplicial homology transformers. The default is always ``True``, which implies a breaking change in the case of ``CubicalPersistence`` (`#467 <https://github.com/giotto-ai/giotto-tda/pull/467>`_).

Persistence diagrams
--------------------

-  A ``ComplexPolynomial`` feature extraction transformer has been added (`#479 <https://github.com/giotto-ai/giotto-tda/pull/479>`_).
-  A ``NumberOfPoints`` feature extraction transformer has been added (`#496 <https://github.com/giotto-ai/giotto-tda/pull/496>`_).
-  An option to normalize the entropy in ``PersistenceEntropy`` according to a heuristic has been added, and a ``nan_fill_value`` parameter allows to replace any NaN produced by the entropy calculation with a fixed constant (`#450 <https://github.com/giotto-ai/giotto-tda/pull/450>`_).
-  The computations in ``HeatKernel``, ``PersistenceImage`` and in the pairwise distances and amplitudes related to them has been changed to yield the continuum limit when ``n_bins`` tends to infinity; ``sigma`` is now measured in the same units as the filtration parameter and defaults to 0.1 (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).

New ``curves`` subpackage
-------------------------

A new ``curves`` subpackage has been added to preprocess, and extract features from, collections of multi-channel curves such as returned by ``BettiCurve``, ``PersistenceLandscape`` and ``Silhouette`` (`#480 <https://github.com/giotto-ai/giotto-tda/pull/480>`_). It contains:

-  A ``StandardFeatures`` transformer that can extract features channel-wise in a generic way.
-  A ``Derivative`` transformer that computes channel-wise derivatives of any order by discrete differences (`#492 <https://github.com/giotto-ai/giotto-tda/pull/492>`_).

New ``metaestimators`` subpackage
---------------------------------

A new ``metaestimator`` subpackage has been added with a ``CollectionTransformer`` meta-estimator which converts any transformer instance into a fit-transformer acting on collections (`#495 <https://github.com/giotto-ai/giotto-tda/pull/495>`_).

Images
------

-  A ``DensityFiltration`` for collections of binary images has been added (`#473 <https://github.com/giotto-ai/giotto-tda/pull/473>`_).
-  ``Padder`` and ``Inverter`` have been extended to greyscale images (`#489 <https://github.com/giotto-ai/giotto-tda/pull/489>`_).

Time series
-----------

-  ``TakensEmbedding`` is now a new transformer acting on collections of time series (`#460 <https://github.com/giotto-ai/giotto-tda/pull/460>`_).
-  The former ``TakensEmbedding`` acting on a single time series has been renamed to ``SingleTakensEmbedding`` transformer, and the internal logic employed in its ``fit`` for computing optimal hyperparameters is now available via a ``takens_embedding_optimal_parameters`` convenience function (`#460 <https://github.com/giotto-ai/giotto-tda/pull/460>`_).
-  The ``_slice_windows`` method of ``SlidingWindow`` has been made public and renamed into ``slice_windows`` (`#460 <https://github.com/giotto-ai/giotto-tda/pull/460>`_).

Graphs
------

-  ``GraphGeodesicDistance`` has been improved as follows (`#422 <https://github.com/giotto-ai/giotto-tda/pull/422>`_):

   -  The new parameters ``directed``, ``unweighted`` and ``method`` have been added.
   -  The rules on the role of zero entries, infinity entries, and non-stored values have been made clearer.
   -  Masked arrays are now supported.

-  A ``mode`` parameter has been added to ``KNeighborsGraph``; as in ``scikit-learn``, it can be set to either ``"distance"`` or ``"connectivity"`` (`#478 <https://github.com/giotto-ai/giotto-tda/pull/478>`_).

-  List input is now accepted by all transformers in ``gtda.graphs``, and outputs are consistently either lists or 3D arrays (`#478 <https://github.com/giotto-ai/giotto-tda/pull/478>`_).

-  Sparse matrices returned by ``KNeighborsGraph`` and ``TransitionGraph`` now have int dtype (0-1 adjacency matrices), and are not necessarily symmetric (`#478 <https://github.com/giotto-ai/giotto-tda/pull/478>`_).

Mapper
------

-  Pullback cover set labels and partial cluster labels have been added to Mapper node hovertexts (`#445 <https://github.com/giotto-ai/giotto-tda/pull/445>`_).

-  The functionality of ``Nerve`` and ``make_mapper_pipeline`` has been greatly extended (`#447 <https://github.com/giotto-ai/giotto-tda/pull/447>`_ and `#456 <https://github.com/giotto-ai/giotto-tda/pull/456>`_):

   -  Node and edge metadata are now accessible in output ``igraph.Graph`` objects by means of the ``VertexSeq`` and ``EdgeSeq`` attributes ``vs`` and ``es`` (respectively). Graph-level dictionaries are no longer used.
   -  Available node metadata can be accessed by ``graph.vs[attr_name]`` where for ``attr_name`` is one of ``"pullback_set_label"``, ``"partial_cluster_label"``, or ``"node_elements"``.
   -  Sizes of intersections are automatically stored as edge weights, accessible by ``graph.es["weight"]``.
   -  A ``"store_intersections"`` keyword argument has been added to ``Nerve`` and ``make_mapper_pipeline`` to allow to store the indices defining node intersections as edge attributes, accessible via ``graph.es["edge_elements"]``.
   -  A ``contract_nodes`` optional parameter has been added to both ``Nerve`` and ``make_mapper_pipeline``; nodes which are subsets of other nodes are thrown away from the graph when this parameter is set to ``True``.
   -  A ``graph_`` attribute is stored during ``Nerve.fit``.

-  Two of the ``Nerve`` parameters (``min_intersection`` and the new ``contract_nodes``) are now available in the widgets generated by ``plot_interactive_mapper_graph``, and the layout of these widgets has been improved (`#456 <https://github.com/giotto-ai/giotto-tda/pull/456>`_).

-  ``ParallelClustering`` and ``Nerve`` have been exposed in the documentation and in ``gtda.mapper``'s ``__init__`` (`#447 <https://github.com/giotto-ai/giotto-tda/pull/447>`_).

Plotting
--------

-  A ``plot_params`` kwarg is available in plotting functions and methods throughout to allow user customisability of output figures. The user must pass a dictionary with keys ``"layout"`` and/or ``"trace"`` (or ``"traces"`` in some cases) (`#441 <https://github.com/giotto-ai/giotto-tda/pull/441>`_).
-  Several plots produced by ``plot`` class methods now have default titles (`#453 <https://github.com/giotto-ai/giotto-tda/pull/453>`_).
-  Infinite deaths are now plotted by ``plot_diagrams`` (`#461 <https://github.com/giotto-ai/giotto-tda/pull/461>`_).
-  Possible multiplicities of persistence pairs in persistence diagram plots are now indicated in the hovertext (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  ``plot_heatmap`` now accepts boolean array input (`#444 <https://github.com/giotto-ai/giotto-tda/pull/444>`_).

New tutorials and examples
--------------------------

The following new tutorials have been added:

-  `Topology of time series <https://giotto-ai.github.io/gtda-docs/0.3.0/notebooks/time_series_classification.html>`_, which explains the theory of the Takens time-delay embedding and its use with persistent homology, demonstrates the new ``API`` of several components in ``gtda.time_series``, and shows how to construct time series *classification* pipelines in ``giotto-tda`` by partially reproducing `arXiv:1910:08245 <https://arxiv.org/abs/1910.08245>`_.
-  `Topology in time series forecasting <https://giotto-ai.github.io/gtda-docs/0.3.0/notebooks/time_series_forecasting.html>`_, which explains how to set up time series *forecasting* pipelines in ``giotto-tda`` via ``TransformerResamplerMixin``s and the ``giotto-tda`` ``Pipeline`` class.
-  `Topological feature extraction from graphs <https://giotto-ai.github.io/gtda-docs/0.3.0/notebooks/persistent_homology_graphs.html>`_, which explains what the features extracted from directed or undirected graphs by ``VietorisRipsPersistence``, ``SparseRipsPersistence`` and ``FlagserPersistence`` are.
-  `Classifying handwritten digits <https://giotto-ai.github.io/gtda-docs/0.3.0/notebooks/MNIST_classification.html>`_, which presents a fully-fledged machine learning pipeline in which cubical persistent homology is applied to the classification of handwritten images from he MNIST dataset, partially reproducing `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

Utils
-----

-  A ``check_collection`` input validation function has been added (`#491 <https://github.com/giotto-ai/giotto-tda/pull/491>`_).
-  ``validate_params`` now accepts ``"in"`` and ``"of"`` keys simultaneously in the ``references`` dictionaries, with ``"in"`` used for non-list-like types and ``"of"`` otherwise (`#502 <https://github.com/giotto-ai/giotto-tda/pull/502>`_).

Installation improvements
-------------------------

-  ``pybind11`` is now treated as a standard git submodule in the developer installation (`#459 <https://github.com/giotto-ai/giotto-tda/pull/459>`_).
-  ``pandas`` is now part of the testing requirements when intalling from source (`#508 <https://github.com/giotto-ai/giotto-tda/pull/508>`_).

Bug Fixes
=========

-  A bug has been fixed which could lead to features with negative lifetime in persistent homology transformers when ``infinity_values`` was set too low (`#339 <https://github.com/giotto-ai/giotto-tda/pull/339>`_).
-  By relying on ``scipy``'s ``shortest_path`` instead of ``scikit-learn``'s ``graph_shortest_path``, some errors in computing ``GraphGeodesicDistance`` (e.g. when som edges are zero) have been fixed (`#422 <https://github.com/giotto-ai/giotto-tda/pull/422>`_).
-  A bug in the handling of COO matrices by the ``ripser`` interface has been fixed (`#465 <https://github.com/giotto-ai/giotto-tda/pull/465>`_).
-  A bug which led to the incorrect handling of the ``homology_dimensions`` parameter in ``Filtering`` has been fixed (`#439 <https://github.com/giotto-ai/giotto-tda/pull/439>`_).
-  An issue with the use of ``joblib.Parallel``, which led to errors when attempting to run ``HeatKernel``, ``PersistenceImage``, and the corresponding amplitudes and distances on large datasets, has been fixed (`#428 <https://github.com/giotto-ai/giotto-tda/pull/428>`_ and `#481 <https://github.com/giotto-ai/giotto-tda/pull/481>`_).
-  A bug leading to plots of persistence diagrams not showing points with negative births or deaths has been fixed, as has a bug with the computation of the range to be shown in the plot (`#437 <https://github.com/giotto-ai/giotto-tda/pull/437>`_).
-  A bug in the handling of persistence pairs with negative death values by ``Filtering`` has been fixed (`#436 <https://github.com/giotto-ai/giotto-tda/pull/436>`_).
-  A bug in the handling of ``homology_dimension_ix`` (now renamed to ``homology_dimension_idx``) in the ``plot`` methods of ``HeatKernel`` and ``PersistenceImage`` has been fixed (`#452 <https://github.com/giotto-ai/giotto-tda/pull/452>`_).
-  A bug in the labelling of axes in ``HeatKernel`` and ``PersistenceImage`` plots has ben fixed (`#453 <https://github.com/giotto-ai/giotto-tda/pull/453>`_ and `#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  ``PersistenceLandscape`` plots now show all homology dimensions, instead of just the first (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  A bug in the computation of amplitudes and pairwise distances based on persistence images has been fixed (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  ``Silhouette`` now does not create NaNs when a subdiagram is trivial (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  ``CubicalPersistence`` now does not create pairs with negative persistence when ``infinity_values`` is set too low (`#467 <https://github.com/giotto-ai/giotto-tda/pull/467>`_).
-  Warnings are no longer thrown by ``KNeighborsGraph`` when ``metric="precomputed"`` (`#506 <https://github.com/giotto-ai/giotto-tda/pull/506>`_).
-  A bug in ``Labeller.resample`` affecting cases in which ``n_steps_future >= size - 1``, has been fixed (`#460 <https://github.com/giotto-ai/giotto-tda/pull/460>`_).
-  A bug in ``validate_params``, affecting the case of tuples of allowed types, has been fixed (`#502 <https://github.com/giotto-ai/giotto-tda/pull/502>`_).

Backwards-Incompatible Changes
==============================

-  The minimum required versions from most of the dependencies have been bumped. The updated dependencies are ``numpy >= 1.19.1``, ``scipy >= 1.5.0``, ``joblib >= 0.16.0``, ``scikit-learn >= 0.23.1``, ``python-igraph >= 0.8.2``, ``plotly >= 4.8.2``, and ``pyflagser >= 0.4.1`` (`#457 <https://github.com/giotto-ai/giotto-tda/pull/457>`_).
- ``GraphGeodesicDistance`` now returns either lists or 3D dense ndarrays for compatibility with the homology transformers -  By relying on ``scipy``'s ``shortest_path`` instead of ``scikit-learn``'s ``graph_shortest_path``, some errors in computing ``GraphGeodesicDistance`` (e.g. when som edges are zero) have been fixed (`#422 <https://github.com/giotto-ai/giotto-tda/pull/422>`_).
-  The output of ``PairwiseDistance`` has been transposed to match ``scikit-learn`` convention ``(n_samples_transform, n_samples_fit)`` (`#420 <https://github.com/giotto-ai/giotto-tda/pull/420>`_).
-  ``plot`` class methods now return figures instead of showing them (`#441 <https://github.com/giotto-ai/giotto-tda/pull/441>`_).
-  Mapper node and edge attributes are no longer stored as graph-level dictionaries, ``"node_id"`` is no longer an available node attribute, and the attributes ``nodes_`` and ``edges_`` previously stored by ``Nerve.fit`` have been removed in favour of a ``graph_`` attribute (`#447 <https://github.com/giotto-ai/giotto-tda/pull/447>`_).
-  The ``homology_dimension_ix`` parameter available in some transformers in ``gtda.diagrams`` has been renamed to ``homology_dimensions_idx`` (`#452 <https://github.com/giotto-ai/giotto-tda/pull/452>`_).
-  The base of the logarithm used by ``PersistenceEntropy`` is now 2 instead of *e*, and NaN values are replaced with -1 instead of 0 by default (`#450 <https://github.com/giotto-ai/giotto-tda/pull/450>`_ and `#474 <https://github.com/giotto-ai/giotto-tda/pull/474>`_).
-  The outputs of ``PersistenceImage``, ``HeatKernel`` and of the pairwise distances and amplitudes based on them is now different due to the improvements described above.
-  Weights are no longer stored in the ``effective_metric_params_`` attribute of ``PairwiseDistance``, ``Amplitude`` and ``Scaler`` objects when the metric is persistence-image–based; only the weight function is (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  The ``homology_dimensions_`` attributes of several transformers have been converted from lists to tuples. When possible, homology dimensions stored as parts of attributes are now presented as ints (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  ``gaussian_filter`` (used to make heat– and persistence-image–based representations/pairwise distances/amplitudes) is now called with ``mode="constant"`` instead of ``"reflect"`` (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  The default value of ``order`` in ``Amplitude`` has been changed from ``2.`` to ``None``, giving vector instead of scalar features (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  The meaning of the default ``None`` for ``weight_function`` in ``PersistenceImage`` (and in ``Amplitude`` and ``PairwiseDistance`` when ``metric="persistence_image"``) has been changed from the identity function to the function returning a vector of ones (`#454 <https://github.com/giotto-ai/giotto-tda/pull/454>`_).
-  Due to the updates in the GUDHI components, some of the bindings and Python interfaces to the GUDHI C++ components in ``gtda.externals`` have changed (`#468 <https://github.com/giotto-ai/giotto-tda/pull/468>`_).
-  ``Labeller.transform`` now returns a 1D array instead of a column array (`#475 <https://github.com/giotto-ai/giotto-tda/pull/475>`_).
-  ``PersistenceLandscape`` now returns 3D arrays instead of 4D ones, for compatibility with the new ``curves`` subpackage (`#480 <https://github.com/giotto-ai/giotto-tda/pull/480>`_).
-  By default, ``CubicalPersistence`` now removes one infinite bar in H0 (`#467 <https://github.com/giotto-ai/giotto-tda/pull/467>`_, and see above).
-  The former ``width`` parameter in ``SlidingWindow`` and ``Labeller`` has been replaced with a more intuitive ``size`` parameter. The relation between the two is: ``size = width + 1`` (`#460 <https://github.com/giotto-ai/giotto-tda/pull/460>`_).
-  ``clusterer`` is now a required parameter in ``ParallelClustering`` (`#508 <https://github.com/giotto-ai/giotto-tda/pull/508>`_).
-  The ``max_fraction`` parameter in ``FirstSimpleGap`` and ``FirstHistogramGap`` now indicates the floor of ``max_fraction * n_samples``; its default value has been changed from ``None`` to ``1`` (`#412 <https://github.com/giotto-ai/giotto-tda/pull/412>`_).

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo, Guillaume Tauzin, Julian Burella Pérez, Wojciech Reise, Lewis Tunstall, Nick Sale, and Anibal Medina-Mardones.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

*************
Release 0.2.2
*************

Major Features and Improvements
===============================

-  The documentation for ``gtda.mapper.utils.decorators.method_to_transform`` has been improved.
-  A table of contents has been added to the theory glossary.
-  The theory glossary has been restructured by including a section titled "Analysis". Entries for l^p norms, L^p norms and heat vectorization have been added.
-  The project's Azure CI for Windows versions has been sped-up by ensuring that the locally installed boost version is detected.
-  Several python bindings to external code from GUDHI, ripser.py and Hera have been made public: specifically, ``from gtda.externals import *`` now gives power users access to:

   -  ``bottleneck_distance``,
   -  ``wasserstein_distance``,
   -  ``ripser``,
   -  ``SparseRipsComplex``,
   -  ``CechComplex``,
   -  ``CubicalComplex``,
   -  ``PeriodicCubicalComplex``,
   -  ``SimplexTree``,
   -  ``WitnessComplex``,
   - ``StrongWitnessComplex``.

   However, these functionalities are still undocumented.
-  The ``gtda.mapper.visualisation`` and ``gtda.mapper.utils._visualisation`` modules have been thoroughly refactored to improve code clarity, add functionality, change behaviour and fix bugs. Specifically, in figures generated by both ``plot_static_mapper_graph`` and ``plot_interactive_mapper_graph``:

   -  The colorbar no longer shows values rescaled to the interval [0, 1]. Instead, it always shows the true range of node summary statistics.
   -  The values of the node summary statistics are now displayed in the hovertext boxes. A a new keyword argument ``n_sig_figs`` controls their rounding (3 is the default).
   -  ``plotly_kwargs`` has been renamed to ``plotly_params`` (see "Backwards-Incompatible Changes" below).
   -  The dependency on ``matplotlib``'s ``rgb2hex`` and ``get_cmap`` functions has been removed. As no other component in ``giotto-tda`` required ``matplotlib``, the dependency on this library has been removed completely.
   -  A ``node_scale`` keyword argument has been added which can be used to controls the size of nodes (see "Backwards-Incompatible Changes" below).
   -  The overall look of Mapper graphs has been improved by increasing the opacity of node colors so that edges do not hide them, and by reducing the thickness of marker lines.
   
   Furthermore, a ``clone_pipeline`` keyword argument has been added to ``plot_interactive_mapper_graph``, which when set to ``False`` allows the user to mutate the input pipeline via the interactive widget. 

-  The docstrings of ``plot_static_mapper_graph``, ``plot_interactive_mapper_graph`` and ``make_mapper_pipeline`` have been improved.

Bug Fixes
=========

-  A CI bug introduced by an update to the XCode compiler installed on the Azure Mac machines has been fixed.
-  A bug afflicting Mapper colors, which was due to an incorrect rescaling to [0, 1], has been fixed.

Backwards-Incompatible Changes
==============================

-  The keyword parameter ``plotly_kwargs`` in ``plot_static_mapper_graph`` and ``plot_interactive_mapper_graph`` has been renamed to ``plotly_params`` and has now slightly different specifications. A new logic controls how the information contained in ``plotly_params`` is used to update plotly figures.
-  The function ``get_node_sizeref`` in ``gtda.mapper.utils.visualization`` has been hidden by renaming it to ``_get_node_sizeref``. Its main intended use is subsumed by the new ``node_scale`` parameter of ``plot_static_mapper_graph`` and ``plot_interactive_mapper_graph``.

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo, Julian Burella Pérez, Anibal Medina-Mardones, Wojciech Reise and Guillaume Tauzin.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

*************
Release 0.2.1
*************

Major Features and Improvements
===============================

-  The theory glossary has been improved to include the notions of vectorization, kernel and amplitude for persistence diagrams.
-  The ``ripser`` function in ``gtda.externals.python.ripser_interface`` no longer uses scikit-learn's ``pairwise_distances`` when
   ``metric`` is ``'precomputed'``, thus allowing square arrays with negative entries or infinities to be passed.
-  ``check_point_clouds`` in ``gtda.utils.validation`` now checks for square array input when the input should be a collection of
   distance-type matrices. Warnings guide the user to correctly setting the ``distance_matrices`` parameter. ``force_all_finite=False``
   no longer means accepting NaN input (only infinite input is accepted).
-  ``VietorisRipsPersistence`` in ``gtda.homology.simplicial`` no longer masks out infinite entries in the input to be fed to
   ``ripser``.
-  The docstrings for ``check_point_clouds`` and ``VietorisRipsPersistence`` have been improved to reflect these changes and the
   extra level of generality for ``ripser``.

Bug Fixes
=========

-  The variable used to indicate the location of Boost headers has been renamed from ``Boost_INCLUDE_DIR`` to ``Boost_INCLUDE_DIRS``
   to address developer installation issues in some Linux systems.

Backwards-Incompatible Changes
==============================

-  The keyword parameter ``distance_matrix`` in ``check_point_clouds`` has been renamed to ``distance_matrices``.

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo, Anibal Medina-Mardones, Julian Burella Pérez, Guillaume Tauzin, and Wojciech Reise.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

*************
Release 0.2.0
*************

Major Features and Improvements
===============================

This is a major release which substantially broadens the scope of ``giotto-tda`` and introduces several improvements.
The library's documentation has been greatly improved and is now hosted `via GitHub pages <https://giotto-ai.github.io/gtda-docs/>`_.
It includes rendered jupyter notebooks from the repository's ``examples`` folder, as well as an improved theory glossary,
more detailed installation instructions, improved guidelines for contributing, and an FAQ.

Plotting functions and plotting API
-----------------------------------

This version introduces built-in plotting capabilities to ``giotto-tda``. These come in the form of:

-  a new ``plotting`` subpackage populated with plotting functions for common data structures;
-  a new ``PlotterMixin`` and a class-level plotting API based on newly introduced ``plot``, ``transform_plot`` and
   ``fit_transform_plot`` methods which are now available in several of ``giotto-tda``'s transformers.

Changes and additions to ``gtda.homology``
------------------------------------------

The internal structure of this subpackage has been changed. ``ConsistentRescaling`` has been moved to a new ``point_clouds``
subpackage (see below), and ``gtda.homology`` no longer contains a ``point_clouds`` submodule. Instead, it contains two
submodules, ``simplicial`` and ``cubical``. ``simplicial`` contains the ``VietorisRipsPersistence`` class as well as the
following new classes:

-  ``SparseRipsPersistence``,
-  ``EuclideanCechPersistence``.

The ``cubical`` submodule contains ``CubicalPersistence``, a new class for computing persistent homology of filtered cubical
complexes such as those coming from 2D or 3D greyscale images.

New ``images`` subpackage
-------------------------

The new ``gtda.images`` subpackage contains classes which, together with ``gtda.homology.CubicalPersistence``, extend
the capabilities of ``giotto-tda`` to computer vision, by handling input representing binary or greyscale 2D/3D images
represented as arrays.

The classes in ``gtda.images.filtrations`` are responsible for converting binary image input into greyscale images in a
variety of ways. The greyscale output can then be fed to ``gtda.homology.CubicalPersistence`` to extract topological
signatures in the form of persistence diagrams. These classes are:

-  ``HeightFiltration``,
-  ``RadialFiltration``,
-  ``DilationFiltration``,
-  ``ErosionFiltration``,
-  ``SignedDistanceFiltration``.

The classes in ``gtda.images.preprocessing`` perform a variety of preprocessing steps on either binary or greyscale image
input, as well as conversion to point cloud format. They are:

-  ``Binarizer``,
-  ``Inverter``,
-  ``Padder``,
-  ``ImageToPointCloud``.

New ``point_clouds`` subpackage
-------------------------------

``ConsistentRescaling`` is no longer placed in ``gtda.homology``. Instead, it is now in a ``point_clouds`` subpackage
containing classes which process or modify the geometry of point cloud data. ``gtda.point_clouds`` also contains the new
class ``ConsecutiveRescaling``, written with time series applications in mind.

List of point cloud input
-------------------------

All classes in the ``homology`` subpackage (``VietorisRipsPersistence``, ``SparseRipsPersistence``, and ``EuclideanCechPersistence``)
can now take as inputs to the ``fit`` and ``transform`` methods lists of 2D arrays instead of simply 3D arrays. In this
way, collections of point clouds with varying numbers of points can be processed.

Changes and additions to ``gtda.diagrams``
------------------------------------------

The ``diagrams`` subpackage contains the following new classes:

-  ``PersistenceImage``
-  ``Silhouette``

Additionally, the subpackage has been reorganised as follows:

-  The ``features`` submodule now only contains the *scalar* feature generation classes ``Amplitude`` (moved there from
   ``distance``) and ``PersistenceEntropy``.
-  Classes which produce *vector* representations from persistence diagrams have been moved to the new ``representations``
   submodule.

Changes and additions to ``gtda.utils``
---------------------------------------

-  ``validate_params`` has been thoroughly refactored, documented and exposed for the benefit of developers.
-  ``check_diagrams`` has been modified, documented and exposed for the benefit of developers.
-  The new ``check_point_clouds`` performs validation of inputs consisting of collections of point clouds of distance
   matrices. It accepts both lists of 2D ndarrays and 3D ndarrays, and is used in the ``fit`` and ``transform``
   methods of classes in ``gtda.homology.simplicial`` to allow for list input (see above).

External modules and HPC improvements
-------------------------------------

A substantial effort has been put in improving the quality of the high-performance components contained in ``gtda.externals``.
The end result is a cleaner packaging as well as faster execution of C++ functions due to improved bindings. In particular:

-  Two binaries are now shipped for ``ripser``, one of them being optimised for calculations with mod 2 coefficients.
-  Recent improvements by the authors of the ``hera`` C++ library have been integrated in ``giotto-tda``.
-  Compiler optimisations for Windows-based systems have been added.
-  The integration of ``pybind11`` has been improved and several issues arising with ``CMake`` and ``boost`` during
   developer installations have been addressed.

Bug Fixes
=========

-  Fixed a bug with ``TakensEmbedding``'s algorithm for search of optimal parameters.
-  Inconsistencies in between the meaning of "bottleneck amplitude" in the theory and in the code have been ironed out.
   The code has been modified to agree with the theory glossary. The outputs of the ``gtda.diagrams`` classes
   ``Amplitude``, ``Scaler`` and ``Filtering`` is affected.
-  Fixed bugs affecting color normalization in Mapper graph plots.

Backwards-Incompatible Changes
==============================

-  Python 3.5 is no longer supported.
-  Mac OS X versions below 10.14 are no longer supported by the wheels shipped via PyPI.
-  ``ConsistentRescaling`` is no longer found in ``gtda.homology`` and is now part of ``gtda.point_clouds``.
-  The outputs of the ``gtda.diagrams`` classes ``Amplitude``, ``Scaler`` and ``Filtering`` have changed due to sqrt(2)
   factors (see Bug Fixes).
-  The ``meta_transformers`` module has been removed.
-  The ``plotting`` module has been removed from the ``examples`` folder of the repository.

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo, Guillaume Tauzin, Wojciech Reise, Julian Burella Pérez, Roman Yurchak, Lewis Tunstall, Anibal Medina-Mardones, and Adélie Garin.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of
inspiring discussions.

*************
Release 0.1.4
*************

Library name change
===================
The library and GitHub repository have been renamed to ``giotto-tda``! While the
new name is meant to better convey the library's focus on Topology-powered
machine learning and Data Analysis, the commitment to seamless integration with
``scikit-learn`` will remain just as strong and a defining feature of the project.
Concurrently, the main module has been renamed from ``giotto`` to ``gtda`` in this
version. ``giotto-learn`` will remain on PyPI as a legacy package (stuck at v0.1.3)
until we have ensured that users and developers have fully migrated. The new PyPI
package ``giotto-tda`` will start at v0.1.4 for project continuity.

Short summary: install via ::

    python -m pip install -U giotto-tda

and ``import gtda`` in your scripts or notebooks!

Change of license
=================

The license changes from Apache 2.0 to GNU AGPLv3 from this release on.

Major Features and Improvements
===============================
-  Added a ``mapper`` submodule implementing the Mapper algorithm of Singh, Mémoli and Carlsson. The main tools are the
   functions ``make_mapper_pipeline``, ``plot_static_mapper_graph`` and ``plot_interactive_mapper_graph``. The first
   creates an object of class ``MapperPipeline`` which can be fit-transformed to data to create a Mapper graph in the
   form of an ``igraph.Graph`` object (see below). The ``MapperPipeline`` class itself is a simple subclass
   of scikit-learn's ``Pipeline`` which is adapted to the precise structure of the Mapper algorithm, so that a
   ``MapperPipeline`` object can be used as part of even larger scikit-learn pipelines, inside a meta-estimator, in a
   grid search, etc. One also has access to other important features of scikit-learn's ``Pipeline``, such as memory
   caching to avoid unnecessary recomputation of early steps when parameters involved in later steps are changed.
   The clustering step can be parallelised over the pullback cover sets via ``joblib`` -- though this can actually
   *lower* performance in small- and medium-size datasets. A range of pre-defined filter functions are also included,
   as well as covers in one and several dimensions, agglomerative clustering algorithms based on stopping rules to
   create flat cuts, and utilities for making transformers out of callables or out of other classes which have no
   ``transform`` method. ``plot_static_mapper_graph`` allows the user to visualise (in 2D or 3D) the Mapper graph
   arising from fit-transforming a ``MapperPipeline`` to data, and offers a range of colouring options to correlate the
   graph's structure with exogenous or endogenous information. It relies on ``plotly`` for plotting and displaying
   metadata. ``plot_interactive_mapper_graph`` adds interactivity to this, via ``ipywidgets``: specifically, the user
   can fine-tune some parameters involved in the definition of the Mapper pipeline, and observe in real time how the
   structure of the graph changes as a result. In this release, all hyperparameters involved in the covering and
   clustering steps are supported. The ability to fine-tune other hyperparameters will be considered for future versions.
-  Added support for Python 3.8.

Bug Fixes
=========
-  Fixed consistently incorrect documentation for the ``fit_transform`` methods. This has been achieved by introducing a
   class decorator ``adapt_fit_transform_docs`` which is defined in the newly introduced ``gtda.utils._docs.py``.

Backwards-Incompatible Changes
==============================
-  The library name change and the change in the name of the main module ``giotto``
   are important major changes.
-  There are now additional dependencies in the ``python-igraph``, ``matplotlib``, ``plotly``, and ``ipywidgets`` libraries.

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo, Lewis Tunstall, Guillaume Tauzin, Philipp Weiler, Julian Burella Pérez.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

*************
Release 0.1.3
*************

Major Features and Improvements
===============================
None

Bug Fixes
=========
-  Fixed a bug in ``diagrams.Amplitude`` causing the transformed array to be wrongly filled and added adequate test.

Backwards-Incompatible Changes
==============================
None.

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


Release 0.1.2
*************

Major Features and Improvements
===============================
-  Added support for Python 3.5.

Bug Fixes
=========
None.

Backwards-Incompatible Changes
==============================
None.

Thanks to our Contributors
==========================

This release contains contributions from many people:

Matteo Caorsi, Henry Tom (@henrytomsf), Guillaume Tauzin.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

*************
Release 0.1.1
*************

Major Features and Improvements
===============================
-  Improved documentation.
-  Improved features of class ``Labeller``.
-  Improved features of class ``PearsonDissimilarities``.
-  Improved GitHub files.
-  Improved CI.

Bug Fixes
=========
Fixed minor bugs from the first release.

Backwards-Incompatible Changes
==============================
The following class were renamed:
-  class ``PearsonCorrelation`` was renamed to class ``PearsonDissimilarities``

Thanks to our Contributors
==========================

This release contains contributions from many people:

Umberto Lupo, Guillaume Tauzin, Matteo Caorsi, Olivier Morel.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

*************
Release 0.1.0
*************

Major Features and Improvements
===============================

The following submodules where added:

-  ``giotto.homology`` implements transformers to modify metric spaces or generate persistence diagrams.
-  ``giotto.diagrams`` implements transformers to preprocess persistence diagrams or extract features from them.
-  ``giotto.time_series`` implements transformers to preprocess time series or embed them in a higher dimensional space for persistent homology.
-  ``giotto.graphs`` implements transformers to create graphs or extract metric spaces from graphs.
-  ``giotto.meta_transformers`` implements convenience ``giotto.Pipeline`` transformers for direct topological feature generation.
-  ``giotto.utils`` implements hyperparameters and input validation functions.
-  ``giotto.base`` implements a ``TransformerResamplerMixin`` for transformers that have a resample method.
-  ``giotto.pipeline`` extends scikit-learn's module by defining Pipelines that include ``TransformerResamplers``.


Bug Fixes
=========
None

Backwards-Incompatible Changes
==============================
None

Thanks to our Contributors
==========================

This release contains contributions from many people:

Guillaume Tauzin, Umberto Lupo, Philippe Nguyen, Matteo Caorsi, Julian Burella Pérez, Alessio Ghiraldello.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions. In particular, we would like
to thank `Martino Milani <https://github.com/MartMilani/reportPACS>`_, who worked on an early
prototype of a Mapper implementation; although very different from the current one, it
adopted an early form of caching to avoid recomputation in refitting, which was an inspiration
for this implementation.

**************
Release 0.1a.0
**************

Initial release of the library, originally named ``giotto-learn``.
