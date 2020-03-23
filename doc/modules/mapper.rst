:mod:`gtda.mapper`: Mapper
==========================

.. automodule:: gtda.mapper
   :no-members:
   :no-inherited-members:

.. figure:: ../images/mapper_pipeline.svg

Filters
-------

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/filters
   :template: class.rst

   mapper.Projection
   mapper.Eccentricity
   mapper.Entropy

Covers
-------

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/covers
   :template: class.rst

   mapper.OneDimensionalCover
   mapper.CubicalCover

Clustering
----------

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/clustering/
   :template: class.rst

   mapper.FirstSimpleGap
   mapper.FirstHistogramGap

Pipeline
--------

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/pipeline/
   :template: function.rst

   mapper.pipeline.make_mapper_pipeline


.. autosummary::
   :toctree: generated/mapper/pipeline/
   :template: class.rst

   mapper.pipeline.MapperPipeline

Visualization
-------------

.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/visualization
   :template: function.rst

   mapper.visualization.plot_static_mapper_graph
   mapper.visualization.plot_interactive_mapper_graph

Utilities
---------
.. currentmodule:: gtda

.. autosummary::
   :toctree: generated/mapper/utils
   :template: function.rst

   mapper.utils.decorators.method_to_transform
   mapper.utils.pipeline.transformer_from_callable_on_rows