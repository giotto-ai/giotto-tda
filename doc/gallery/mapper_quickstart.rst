.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_gallery_mapper_quickstart.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_gallery_mapper_quickstart.py:


Getting Started with Mapper
===========================

In this notebook we explore a few of the core features included in
giotto-tda’s implementation of the `Mapper
algorithm <https://research.math.osu.edu/tgda/mapperPBG.pdf>`__.

Useful references
-----------------

-  `An introduction to Topological Data Analysis: fundamental and
   practical aspects for data
   scientists <https://arxiv.org/abs/1710.04019>`__
-  `An Introduction to Topological Data Analysis for Physicists: From
   LGM to FRBs <https://arxiv.org/abs/1904.11044>`__

License: AGPLv3
^^^^^^^^^^^^^^^

Import libraries
----------------



.. code-block:: default


    # Data wrangling
    import numpy as np
    import pandas as pd  # Not a requirement of giotto-tda, but is compatible with the gtda.mapper module

    # Data viz
    from plotting import plot_point_cloud

    # TDA magic
    from gtda.mapper import (
        CubicalCover,
        make_mapper_pipeline,
        Projection,
        plot_static_mapper_graph,
        plot_interactive_mapper_graph,
    )
    from gtda.mapper.utils.visualization import set_node_sizeref

    # ML tools
    from sklearn import datasets
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)



Generate and visualise data
---------------------------

As a simple example, let’s generate a two-dimensional point cloud of two
concentric circles. The goal will be to examine how Mapper can be used
to generate a topological graph that captures the salient features of
the data.



.. code-block:: default


    data, _ = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)

    plot_point_cloud(data)



Configure the Mapper pipeline
-----------------------------

Given a dataset :math:`{\cal D}` of points :math:`x \in \mathbb{R}^n`,
the basic steps behind Mapper are as follows:

1. Map :math:`{\cal D}` to a lower-dimensional space using a **filter
   function** $ f: :raw-latex:`\mathbb{R}`^n
   :raw-latex:`\to `:raw-latex:`\mathbb{R}`^m $. Common choices for the
   filter function include projection onto one or more axes via PCA or
   density-based methods. In giotto-tda, you can import a variety of
   filter functions as follows:

.. code:: python

   from gtda.mapper.filter import FilterFunctionName

2. Construct a cover of the filter values
   :math:`{\cal U} = (U_i)_{i\in I}`, typically in the form of a set of
   overlapping intervals which have constant length. As with the filter,
   a choice of cover can be imported as follows:

.. code:: python

   from gtda.mapper.cover import CoverName

3. For each interval :math:`U_i \in {\cal U}` cluster the points in the
   preimage :math:`f^{-1}(U_i)` into sets
   :math:`C_{i,1}, \ldots , C_{i,k_i}`. The choice of clustering
   algorithm can be any of scikit-learn’s `clustering
   methods <https://scikit-learn.org/stable/modules/clustering.html>`__
   or an implementation of agglomerative clustering in giotto-tda:

.. code:: python

   # scikit-learn method
   from sklearn.cluster import ClusteringAlgorithm
   # giotto-tda method
   from gtda.mapper.cluster import FirstSimpleGap

4. Construct the topological graph whose vertices are the cluster sets
   :math:`(C_{i,j})_{i\in I, j \in \{1,\ldots,k_i\}}` and an edge exists
   between two nodes if they share points in common:
   :math:`C_{i,j} \cap C_{k,l} \neq \emptyset`. This step is handled
   automatically by giotto-tda.

These four steps are implemented in the ``MapperPipeline`` object that
mimics the ``Pipeline`` class from scikit-learn. We provide a
convenience function ``make_mapper_pipeline()`` that allows you to pass
the choice of filter function, cover, and clustering algorithm as
arguments. For example, to project our data onto the :math:`x`- and
:math:`y`-axes, we could setup the pipeline as follows:



.. code-block:: default


    # Define filter function - can be any scikit-learn transformer
    filter_func = Projection(columns=[0, 1])
    # Define cover
    cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
    # Choose clustering algorithm - default is DBSCAN
    clusterer = DBSCAN()

    # Configure parallelism of clustering step
    n_jobs = 1

    # Initialise pipeline
    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=False,
        n_jobs=n_jobs,
    )



Visualise the Mapper graph
--------------------------


With the Mapper pipeline at hand, it is now a simple matter to visualise
it. To warm up, let’s examine the graph in two-dimensions using the
default arguments of giotto-tda’s plotting function:



.. code-block:: default


    fig = plot_static_mapper_graph(pipe, data)
    # Display figure
    fig.show(config={"scrollZoom": True})



From the figure we can see that we have captured the salient topological
features of our underlying data, namely two holes!


Configure the coloring of the Mapper graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the nodes of the Mapper graph are colored by the mean value
of the points that belong to a given node. However, in this example it
is more instructive to colour by the :math:`x`- and :math:`y`-axes. This
can be achieved by toggling the ``color_by_columns_dropdown``, which
calculates the coloring for each column in the input data array. At the
same time, let’s configure the choice of colorscale:



.. code-block:: default


    plotly_kwargs = {"node_trace_marker_colorscale": "Blues"}
    fig = plot_static_mapper_graph(
        pipe, data, color_by_columns_dropdown=True, plotly_kwargs=plotly_kwargs
    )
    # Display figure
    fig.show(config={"scrollZoom": True})



In the dropdown menu, the entry ``color_variable`` refers to a
user-defined quantity to color by - by default it is the average value
of the points in each node. In general, one can configure this quantity
to be an array, a scikit-learn transformer, or a list of indices to
select from the data. For example, coloring by a PCA component can be
implemented as follows:



.. code-block:: default


    # Initialise estimator to color graph by
    pca = PCA(n_components=1).fit(data)

    fig = plot_static_mapper_graph(
        pipe, data, color_by_columns_dropdown=True, color_variable=pca
    )
    # Display figure
    fig.show(config={"scrollZoom": True})



Pass a pandas DataFrame as input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


It is also possible to feed ``plot_static_mapper_graph()`` a pandas
DataFrame:



.. code-block:: default


    data_df = pd.DataFrame(data, columns=["x", "y"])
    data_df.head()



Before plotting we need to update the Mapper pipeline to know about the
projection onto the column names. This can be achieved using the
``set_params()`` method as follows:



.. code-block:: default


    pipe.set_params(filter_func=Projection(columns=["x", "y"]))

    fig = plot_static_mapper_graph(pipe, data_df, color_by_columns_dropdown=True)
    # Display figure
    fig.show(config={"scrollZoom": True})



Change the layout algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``plot_static_mapper_graph()`` uses the Kamada–Kawai
algorithm for the layout; however any of the layout algorithms defined
in python-igraph are supported (see
`here <https://igraph.org/python/doc/igraph.Graph-class.html>`__ for a
list of possible layouts). For example, we can switch to the
Fruchterman–Reingold layout as follows:



.. code-block:: default


    # Reset back to numpy projection
    pipe.set_params(filter_func=Projection(columns=[0, 1]))

    fig = plot_static_mapper_graph(
        pipe, data, layout="fruchterman_reingold", color_by_columns_dropdown=True
    )
    # Display figure
    fig.show(config={"scrollZoom": True})



Change the layout dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to visualise the Mapper graph in 3-dimensions by
configuring the ``layout_dim`` argument:



.. code-block:: default


    fig = plot_static_mapper_graph(pipe, data, layout_dim=3, color_by_columns_dropdown=True)
    # Display figure
    fig.show(config={"scrollZoom": True})



Run the Mapper pipeline
-----------------------

Behind the scenes of ``plot_static_mapper_graph()`` is a
``MapperPipeline`` object ``pipe`` that can be used like a typical
scikit-learn estimator. For example, to extract the underlying graph
data structure we can do the following:



.. code-block:: default


    graph = pipe.fit_transform(data)



The resulting graph is an `python-igraph <https://igraph.org/python/>`__
object that contains metadata that is stored in the form of
dictionaries. We can access this data as follows:



.. code-block:: default


    graph["node_metadata"].keys()



Here ``node_id`` is a globally unique node identifier used to construct
the graph, while ``pullback_set_label`` and ``partial_cluster_label``
refer to the interval and cluster sets described above. The
``node_elements`` refers to the indices of our original data that belong
to each node. For example, to find which points belong to the first node
of the graph we can access the desired data as follows:



.. code-block:: default


    node_id, node_elements = (
        graph["node_metadata"]["node_id"],
        graph["node_metadata"]["node_elements"],
    )

    print(
        "Node Id: {}, \nNode elements: {}, \nData points: {}".format(
            node_id[0], node_elements[0], data[node_elements[0]]
        )
    )



The ``node_elements`` are handy for situations when we want to customise
e.g. the size of the node scale. In this example, we use the utility
function ``set_node_sizeref()`` and pass the function as a plotly
argument:



.. code-block:: default


    # Configure scale for node sizes
    plotly_kwargs = {
        "node_trace_marker_sizeref": set_node_sizeref(node_elements, node_scale=30)
    }
    fig = plot_static_mapper_graph(
        pipe,
        data,
        layout_dim=3,
        color_by_columns_dropdown=True,
        plotly_kwargs=plotly_kwargs,
    )
    # Display figure
    fig.show(config={"scrollZoom": True})



The resulting graph is much easier to decipher with the enlarged node
scaling!


Creating custom filter functions
--------------------------------

In some cases, the list of filter functions provided in ``filter.py`` or
scikit-learn may not be sufficient for the task at hand. In such cases,
one can pass any callable to the pipeline that acts *row-wise* on the
input data. For example, we can project by taking the sum of the
:math:`(x,y)` coordinates as follows:



.. code-block:: default


    filter_func = np.sum

    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=True,
        n_jobs=n_jobs,
    )

    fig = plot_static_mapper_graph(pipe, data, plotly_kwargs=None)
    # Display figure
    fig.show(config={"scrollZoom": True})



In general, any callable (i.e. function) that operates **row-wise** can
be passed.


Visualise the 2D Mapper graph interactively
-------------------------------------------

In general, buidling useful Mapper graphs requires some iteration
through the various parameters in the cover and clustering algorithm. To
simplify that process, giotto-tda provides an interactive figure that
can be configured in real-time. If invalid parameters are selected, the
*Show logs* checkbox can be used to see what went wrong.



.. code-block:: default


    pipe = make_mapper_pipeline()

    # Generate interactive plot
    plot_interactive_mapper_graph(pipe, data, color_by_columns_dropdown=True)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_gallery_mapper_quickstart.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: mapper_quickstart.py <mapper_quickstart.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: mapper_quickstart.ipynb <mapper_quickstart.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
