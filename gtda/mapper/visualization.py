"""Static and interactive visualisation functions for Mapper graphs."""
# License: GNU AGPLv3

import logging
import traceback

import numpy as np
import plotly.graph_objects as go
from ipywidgets import widgets, Layout, HTML
from sklearn.base import clone

from .utils._logging import OutputWidgetHandler
from .utils._visualization import (
    _validate_color_kwargs,
    _calculate_graph_data,
    _produce_static_figure,
    _get_column_color_buttons,
    _get_colors_for_vals,
)


def plot_static_mapper_graph(
        pipeline, data, color_data=None, color_features=None,
        node_color_statistic=None, layout="kamada_kawai", layout_dim=2,
        clone_pipeline=True, n_sig_figs=3, node_scale=12, plotly_params=None
        ):
    """Plot Mapper graphs without interactivity on pipeline parameters.

    The output graph is a rendition of the :class:`igraph.Graph` object
    computed by calling the :meth:`fit_transform` method of the
    :class:`~gtda.mapper.pipeline.MapperPipeline` instance `pipeline` on the
    input `data`. The graph's nodes correspond to subsets of elements (rows) in
    `data`; these subsets are clusters in larger portions of `data` called
    "pullback (cover) sets", which are computed by means of the `pipeline`'s
    "filter function" and "cover" and correspond to the differently-colored
    portions in `this diagram <../../../../_images/mapper_pipeline.svg>`_.
    Two clusters from different pullback cover sets can overlap; if they do, an
    edge between the corresponding nodes in the graph may be drawn.

    Nodes are colored according to `color_features` and `node_color_statistic`
    and are sized according to the number of elements they represent. The
    hovertext on each node displays, in this order:

        - a globally unique ID for the node, which can be used to retrieve
          node information from the :class:`igraph.Graph` object, see
          :class:`~gtda.mapper.nerve.Nerve`;
        - the label of the pullback (cover) set which the node's elements
          form a cluster in;
        - a label identifying the node as a cluster within that pullback set;
        - the number of elements of `data` associated with the node;
        - the value of the summary statistic which determines the node's color.

    Parameters
    ----------
    pipeline : :class:`~gtda.mapper.pipeline.MapperPipeline` object
        Mapper pipeline to act onto data.

    data : array-like of shape (n_samples, n_features)
        Data used to generate the Mapper graph. Can be a pandas dataframe.

    color_data : array-like of length n_samples, or None, optional, \
        default: ``None``
        Data to be used to construct node colors in the Mapper graph (according
        to `color_features` and `node_color_statistic`). Must have the same
        length as `data`. ``None`` is the same as passing
        ``numpy.arange(len(data))``.

    color_features : object or None, optional, default: ``None``
        Specifies one or more feature of interest from `color_data` to be used,
        together with `node_color_statistic`, to determine node colors. Ignored
        if `node_color_statistic` is a numpy array.

            1. ``None`` is equivalent to passing `color_data`.
            2. If an object implementing :meth:`transform` or
               :meth:`fit_transform`, or a callable, it is applied to
               `color_data` to generate the features of interest.
            3. If an index or string, or list of indices/strings, it is
               equivalent to selecting a column or subset of columns from
               `color_data`.

    node_color_statistic : None, callable, or ndarray of shape (n_nodes,) or \
        (n_nodes, 1), optional, default: ``None``
        If a callable, node colors will be computed as summary statistics from
        the feature array ``y`` determined by `color_data` and
        `color_features`. Let ``y`` have ``n`` columns (note: 1d feature arrays
        are converted to column vectors). Then, for a node representing a list
        ``I`` of row indices, there will be ``n`` colors, each computed as
        ``node_color_statistic(y[I, i])`` for ``i`` between ``0`` and ``n``.
        ``None`` is equivalent to passing :func:`numpy.mean`. If a numpy array,
        it must have the same length as the number of nodes in the Mapper graph
        and its values are used directly as node colors (`color_features` is
        ignored).

    layout : None, str or callable, optional, default: ``"kamada-kawai"``
        Layout algorithm for the graph. Can be any accepted value for the
        ``layout`` parameter in the :meth:`layout` method of
        :class:`igraph.Graph` [1]_.

    layout_dim : int, default: ``2``
        The number of dimensions for the layout. Can be 2 or 3.

    clone_pipeline : bool, optional, default: ``True``
        If ``True``, the input `pipeline` is cloned before computing the
        Mapper graph to prevent unexpected side effects from in-place
        parameter updates.

    n_sig_figs : int or None, optional, default: ``3``
       If not ``None``, number of significant figures to which to round node
       summary statistics. If ``None``, no rounding is performed.

    node_scale : int or float, optional, default: ``12``
        Sets the scale factor used to determine the rendered size of the
        nodes. Increase for larger nodes. Implements a formula in the
        `Plotly documentation \
        <https://plotly.com/python/bubble-charts/#scaling-the-size-of-bubble\
        -charts>`_.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"node_trace"``, ``"edge_trace"`` and ``"layout"``, and the
        corresponding values should be dictionaries containing keyword
        arguments as would be fed to the :meth:`update_traces` and
        :meth:`update_layout` methods of :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    fig : :class:`plotly.graph_objects.FigureWidget` object
        Figure representing the Mapper graph with appropriate node colouring
        and size.

    Examples
    --------
    Setting a colorscale different from the default one:

    >>> import numpy as np
    >>> np.random.seed(1)
    >>> from gtda.mapper import make_mapper_pipeline, plot_static_mapper_graph
    >>> pipeline = make_mapper_pipeline()
    >>> data = np.random.random((100, 3))
    >>> plotly_params = {"node_trace": {"marker_colorscale": "Blues"}}
    >>> fig = plot_static_mapper_graph(pipeline, data,
    ...                                plotly_params=plotly_params)

    Inspect the composition of a node with "Node ID" displayed as 0 in the
    hovertext:

    >>> graph = pipeline.fit_transform(data)
    >>> graph.vs[0]["node_elements"]
    array([70])

    Write the figure to a file using Plotly:
    >>> fname = "current_figure"
    >>> fig.write_html(fname + ".html")
    >>> fig.write_image(fname + ".svg")  # Requires psutil

    See also
    --------
    MapperInteractivePlotter, plot_interactive_mapper_graph, \
    gtda.mapper.make_mapper_pipeline

    References
    ----------
    .. [1] `igraph.Graph.layout
            <https://igraph.org/python/doc/igraph.Graph-class.html#layout>`_
            documentation.

    """

    # Compute the graph and fetch the indices of points in each node
    _pipeline = clone(pipeline) if clone_pipeline else pipeline

    graph = _pipeline.fit_transform(data)
    (color_data_transformed, column_names_dropdown,
     node_color_statistic) = \
        _validate_color_kwargs(graph, data, color_data, color_features,
                               node_color_statistic, interactive=False)
    edge_trace, node_trace, node_colors_color_features = \
        _calculate_graph_data(
            graph, color_data_transformed, node_color_statistic, layout,
            layout_dim, n_sig_figs, node_scale
            )

    figure = _produce_static_figure(
        edge_trace, node_trace, node_colors_color_features,
        column_names_dropdown, layout_dim, n_sig_figs, plotly_params
        )

    return figure


def plot_interactive_mapper_graph(
        pipeline, data, color_data=None, color_features=None,
        node_color_statistic=None, layout="kamada_kawai", layout_dim=2,
        clone_pipeline=True, n_sig_figs=3, node_scale=12, plotly_params=None
        ):
    """*As of version 0.5.0, we recommend using the object-oriented interface
    provided by :class:`MapperInteractivePlotter` instead of this function.*

    Plot Mapper graphs in a Jupyter session, with interactivity on pipeline
    parameters.

    Extends :func:`~gtda.mapper.visualization.plot_static_mapper_graph` by
    providing functionality to interactively update parameters from the cover,
    clustering and graph construction steps defined in `pipeline`.

    Parameters
    ----------
    pipeline : :class:`~gtda.mapper.pipeline.MapperPipeline` object
        Mapper pipeline to act on to data.

    data : array-like of shape (n_samples, n_features)
        Data used to generate the Mapper graph. Can be a pandas dataframe.

    color_data : array-like of length n_samples, or None, optional, \
        default: ``None``
        Data to be used to construct node colors in the Mapper graph (according
        to `color_features` and `node_color_statistic`). Must have the same
        length as `data`. ``None`` is the same as passing
        ``numpy.arange(len(data))``.

    color_features : object or None, optional, default: ``None``
        Specifies one or more feature of interest from `color_data` to be used,
        together with `node_color_statistic`, to determine node colors.

            1. ``None`` is equivalent to passing `color_data`.
            2. If an object implementing :meth:`transform` or
               :meth:`fit_transform`, or a callable, it is applied to
               `color_data` to generate the features of interest.
            3. If an index or string, or list of indices/strings, it is
               equivalent to selecting a column or subset of columns from
               `color_data`.

    node_color_statistic : None or callable, optional, default: ``None``
        If a callable, node colors will be computed as summary statistics from
        the feature array ``y`` determined by `color_data` and
        `color_features`. Let ``y`` have ``n`` columns (note: 1d feature arrays
        are converted to column vectors). Then, for a node representing a list
        ``I`` of row indices, there will be ``n`` colors, each computed as
        ``node_color_statistic(y[I, i])`` for ``i`` between ``0`` and ``n``.
        ``None`` is equivalent to passing :func:`numpy.mean`.

    layout : None, str or callable, optional, default: ``"kamada-kawai"``
        Layout algorithm for the graph. Can be any accepted value for the
        ``layout`` parameter in the :meth:`layout` method of
        :class:`igraph.Graph` [1]_.

    layout_dim : int, default: ``2``
        The number of dimensions for the layout. Can be 2 or 3.

    clone_pipeline : bool, optional, default: ``True``
        If ``True``, the input `pipeline` is cloned before computing the
        Mapper graph to prevent unexpected side effects from in-place
        parameter updates.

    n_sig_figs : int or None, optional, default: ``3``
       If not ``None``, number of significant figures to which to round node
       summary statistics. If ``None``, no rounding is performed.

    node_scale : int or float, optional, default: ``12``
        Sets the scale factor used to determine the rendered size of the
        nodes. Increase for larger nodes. Implements a formula in the
        `Plotly documentation \
        <plotly.com/python/bubble-charts/#scaling-the-size-of-bubble-charts>`_.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"node_trace"``, ``"edge_trace"`` and ``"layout"``, and the
        corresponding values should be dictionaries containing keyword
        arguments as would be fed to the :meth:`update_traces` and
        :meth:`update_layout` methods of :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    box : :class:`ipywidgets.VBox` object
        A box containing the following widgets: parameters of the clustering
        algorithm, parameters for the covering scheme, a Mapper graph arising
        from those parameters, a validation box, and logs.

    See also
    --------
    MapperInteractivePlotter, plot_static_mapper_graph, \
    gtda.mapper.pipeline.make_mapper_pipeline

    References
    ----------
    .. [1] `igraph.Graph.layout
            <https://igraph.org/python/doc/igraph.Graph-class.html#layout>`_
            documentation.

    """

    plotter = MapperInteractivePlotter(pipeline, data, clone_pipeline)

    return plotter.plot(
        color_data=color_data, color_features=color_features,
        node_color_statistic=node_color_statistic, layout=layout,
        layout_dim=layout_dim, n_sig_figs=n_sig_figs, node_scale=node_scale,
        plotly_params=plotly_params
        )


class MapperInteractivePlotter:
    """Plot Mapper graphs in a Jupyter session, with interactivity on pipeline
    parameters.

    Provides functionality to interactively update parameters from the cover,
    clustering and graph construction steps defined in `pipeline`.
    An interactive widget is produced when calling :meth:`plot`. After
    interacting with the widget, the current state of all outputs which may
    have been altered can be retrieved via one of the attributes listed below.

    Parameters
    ----------
    pipeline : :class:`~gtda.mapper.pipeline.MapperPipeline` object
        Mapper pipeline to act on to data.

    data : array-like of shape (n_samples, n_features)
        Data used to generate the Mapper graph. Can be a pandas dataframe.

    clone_pipeline : bool, optional, default: ``True``
        If ``True``, the input `pipeline` is cloned before computing the
        Mapper graph to prevent unexpected side effects from in-place
        parameter updates.

    Attributes
    ----------
    graph_ : :class:`igraph.Graph` object
        Current state of the graph displayed by the widget.

    pipeline_ : :class:`~gtda.mapper.pipeline.MapperPipeline` object
        Current state of the Mapper pipeline.

    color_features_ : array-like of shape (n_samples, n_features)
        Values of the features of interest for each entry in `data`, as
        produced according to `color_data` and `color_features` when calling
        :meth:`plot`. Not changed by interacting with the widget.

    node_summaries_ : array-like of shape (n_nodes, n_features)
        Current values of the summaries computed for each node and used as
        node colours in the figure. Produced according to
        `node_color_statistic`, see :meth:`plot`.

    figure_ : :class:`plotly.graph_objects.FigureWidget` object
        Current figure representing the Mapper graph with appropriate node
        colouring and size.

    Examples
    --------
    Instantiate the plotter object on a pipeline and data configuration, and
    call :meth:`plot` to display the widget in a Jupyter session:

    >>> import numpy as np
    >>> np.random.seed(1)
    >>> from gtda.mapper import make_mapper_pipeline, MapperInteractivePlotter
    >>> pipeline = make_mapper_pipeline()
    >>> data = np.random.random((100, 3))
    >>> plotter = MapperInteractivePlotter(pipeline, data)
    >>> plotter.plot()

    After interacting with the widget, inspect the composition of a node with
    "Node ID" displayed as 0 in the hovertext:

    >>> plotter.graph_.vs[0]["node_elements"]
    array([70])

    Write the current figure to a file using Plotly:
    >>> fname = "current_figure"
    >>> plotter.fig_.write_html(fname + ".html")
    >>> plotter.fig_.write_image(fname + ".svg")  # Requires psutil

    See also
    --------
    plot_interactive_mapper_graph, plot_static_mapper_graph, \
    gtda.mapper.pipeline.make_mapper_pipeline

    References
    ----------
    .. [1] `igraph.Graph.layout
            <https://igraph.org/python/doc/igraph.Graph-class.html#layout>`_
            documentation.

    """

    def __init__(self, pipeline, data, clone_pipeline=True):
        self.pipeline = pipeline
        self.data = data
        self.clone_pipeline = clone_pipeline

    def plot(self, color_data=None, color_features=None,
             node_color_statistic=None, layout="kamada_kawai", layout_dim=2,
             n_sig_figs=3, node_scale=12, plotly_params=None):
        """ Produce the interactive Mapper widget.

        Parameters
        ----------
        color_data : array-like of length n_samples, or None, optional, \
            default: ``None``
            Data to be used to construct node colors in the Mapper graph
            (according to `color_features` and `node_color_statistic`). Must
            have the same length as `data`. ``None`` is the same as passing
            ``numpy.arange(len(data))``.

        color_features : object or None, optional, default: ``None``
            Specifies one or more feature of interest from `color_data` to be
            used, together with `node_color_statistic`, to determine node
            colors.

                1. ``None`` is equivalent to passing `color_data`.
                2. If an object implementing :meth:`transform` or
                   :meth:`fit_transform`, or a callable, it is applied to
                   `color_data` to generate the features of interest.
                3. If an index or string, or list of indices/strings, it is
                   equivalent to selecting a column or subset of columns from
                   `color_data`.

        node_color_statistic : None or callable, optional, default: ``None``
            If a callable, node colors will be computed as summary statistics
            from the feature array ``y`` determined by `color_data` and
            `color_features`. Let ``y`` have ``n`` columns (note: 1d feature
            arrays are converted to column vectors). Then, for a node
            representing a list ``I`` of row indices, there will be ``n``
            colors, each computed as ``node_color_statistic(y[I, i])`` for
            ``i`` between ``0`` and ``n``.

        layout : None, str or callable, optional, default: ``"kamada-kawai"``
            Layout algorithm for the graph. Can be any accepted value for the
            ``layout`` parameter in the :meth:`layout` method of
            :class:`igraph.Graph` [1]_.

        layout_dim : int, default: ``2``
            The number of dimensions for the layout. Can be 2 or 3.

        n_sig_figs : int or None, optional, default: ``3``
           If not ``None``, number of significant figures to which to round
           node summary statistics. If ``None``, no rounding is performed.

        node_scale : int or float, optional, default: ``12``
            Sets the scale factor used to determine the rendered size of the
            nodes. Increase for larger nodes. Implements a formula in the
            `Plotly documentation \
            <plotly.com/python/bubble-charts/#scaling-the-size-of-bubble-charts>`_.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"node_trace"``, ``"edge_trace"`` and ``"layout"``, and the
            corresponding values should be dictionaries containing keyword
            arguments as would be fed to the :meth:`update_traces` and
            :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        box : :class:`ipywidgets.VBox` object
            A box containing the following widgets: parameters of the
            clustering algorithm, parameters for the covering scheme, a Mapper
            graph arising from those parameters, a validation box, and logs.

        """
        # Clone pipeline to avoid side effects from in-place parameter changes
        if self.clone_pipeline:
            self._pipeline = clone(self.pipeline)
        else:
            self._pipeline = self.pipeline

        def get_widgets_per_param(params):
            for key, value in params.items():
                style = {'description_width': 'initial'}
                description = key.split("__")[1] if "__" in key else key
                if isinstance(value, float):
                    yield (key, widgets.FloatText(
                        value=value,
                        step=0.05,
                        description=description,
                        continuous_update=False,
                        disabled=False,
                        layout=Layout(width="90%"),
                        style=style
                    ))
                elif isinstance(value, bool):
                    yield (key, widgets.ToggleButton(
                        value=value,
                        description=description,
                        disabled=False,
                        layout=Layout(width="90%"),
                        style=style
                    ))
                elif isinstance(value, int):
                    yield (key, widgets.IntText(
                        value=value,
                        step=1,
                        description=description,
                        continuous_update=False,
                        disabled=False,
                        layout=Layout(width="90%"),
                        style=style
                    ))
                elif isinstance(value, str):
                    yield (key, widgets.Text(
                        value=value,
                        description=description,
                        continuous_update=False,
                        disabled=False,
                        layout=Layout(width="90%"),
                        style=style
                    ))

        def on_parameter_change(change):
            handler.clear_logs()
            try:
                for param, value in cover_params.items():
                    if isinstance(value, (int, float, str)):
                        self._pipeline.set_params(
                            **{param: cover_params_widgets[param].value}
                        )
                for param, value in cluster_params.items():
                    if isinstance(value, (int, float, str)):
                        self._pipeline.set_params(
                            **{param: cluster_params_widgets[param].value}
                        )
                for param, value in nerve_params.items():
                    if isinstance(value, (int, bool)):
                        self._pipeline.set_params(
                            **{param: nerve_params_widgets[param].value}
                        )

                logger.info("Updating figure...")
                with self._figure.batch_update():
                    self._graph = self._pipeline.fit_transform(self.data)
                    (edge_trace, node_trace,
                     self._node_colors_color_features) = \
                        _calculate_graph_data(
                            self._graph, self._color_data_transformed,
                            node_color_statistic, layout, layout_dim,
                            n_sig_figs, node_scale
                        )
                    if colorscale_for_hoverlabel is not None:
                        min_col, max_col = \
                            np.min(self._node_colors_color_features[:, 0]), \
                            np.max(self._node_colors_color_features[:, 0])
                        hoverlabel_bgcolor = _get_colors_for_vals(
                            self._node_colors_color_features[:, 0],
                            min_col, max_col, colorscale_for_hoverlabel
                            )
                        self._figure.update_traces(
                            hoverlabel_bgcolor=hoverlabel_bgcolor,
                            selector={"name": "node_trace"}
                            )

                    self._figure.update_traces(
                        x=node_trace.x,
                        y=node_trace.y,
                        marker_color=node_trace.marker.color,
                        marker_size=node_trace.marker.size,
                        marker_sizeref=node_trace.marker.sizeref,
                        hovertext=node_trace.hovertext,
                        **({"z": node_trace.z} if layout_dim == 3 else dict()),
                        selector={"name": "node_trace"}
                    )
                    self._figure.update_traces(
                        x=edge_trace.x,
                        y=edge_trace.y,
                        **({"z": edge_trace.z} if layout_dim == 3 else dict()),
                        selector={"name": "edge_trace"}
                    )

                    # Update color by column buttons if relevant
                    if self._node_colors_color_features.shape[1] > 1:
                        hovertext_color_features = node_trace.hovertext
                        column_color_buttons = _get_column_color_buttons(
                            self._node_colors_color_features,
                            hovertext_color_features,
                            colorscale_for_hoverlabel, n_sig_figs,
                            column_names_dropdown
                        )

                        button_height = 1.1
                        self._figure.update_layout(
                            updatemenus=[
                                go.layout.Updatemenu(
                                    buttons=column_color_buttons,
                                    direction="down",
                                    pad={"r": 10, "t": 10},
                                    showactive=True,
                                    x=0.11,
                                    xanchor="left",
                                    y=button_height,
                                    yanchor="top"
                                )
                            ])

                valid.value = True
            except Exception:
                exception_data = traceback.format_exc().splitlines()
                logger.exception(exception_data[-1])
                valid.value = False

        def observe_widgets(params, widgets):
            for param, value in params.items():
                if isinstance(value, (int, float, str)):
                    widgets[param].observe(on_parameter_change, names="value")

        # Define output widget to capture logs
        out = widgets.Output()

        @out.capture()
        def click_box(change):
            if logs_box.value:
                out.clear_output()
                handler.show_logs()
            else:
                out.clear_output()

        # Initialise logging
        logger = logging.getLogger(__name__)
        handler = OutputWidgetHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Initialise cover, cluster and nerve dictionaries of parameters and
        # widgets
        mapper_params_items = self._pipeline.get_mapper_params().items()
        cover_params = {key: value for key, value in mapper_params_items
                        if key.startswith("cover__")}
        cover_params_widgets = dict(get_widgets_per_param(cover_params))
        cluster_params = {key: value for key, value in mapper_params_items
                          if key.startswith("clusterer__")}
        cluster_params_widgets = dict(get_widgets_per_param(cluster_params))
        nerve_params = {key: value for key, value in mapper_params_items
                        if key in ["min_intersection", "contract_nodes"]}
        nerve_params_widgets = dict(get_widgets_per_param(nerve_params))

        # Initialise widgets for validating input parameters of pipeline
        valid = widgets.Valid(
            value=True,
            description="Valid parameters",
            style={"description_width": "100px"},
        )

        # Initialise widget for showing the logs
        logs_box = widgets.Checkbox(
            description="Show logs: ",
            value=False,
            indent=False
        )

        # Initialise figure with initial pipeline and config
        self._graph = self._pipeline.fit_transform(self.data)
        (self._color_data_transformed, column_names_dropdown,
         node_color_statistic) = \
            _validate_color_kwargs(self._graph, self.data, color_data,
                                   color_features, node_color_statistic,
                                   interactive=True)
        edge_trace, node_trace, self._node_colors_color_features = \
            _calculate_graph_data(
                self._graph, self._color_data_transformed,
                node_color_statistic, layout, layout_dim, n_sig_figs,
                node_scale
            )

        self._figure = _produce_static_figure(
            edge_trace, node_trace, self._node_colors_color_features,
            column_names_dropdown, layout_dim, n_sig_figs, plotly_params
        )

        colorscale_for_hoverlabel = None
        if layout_dim == 3:
            # In plot_static_mapper_graph, hoverlabel bgcolors are set to white
            # if something goes wrong in computing them according to the
            # colorscale.
            is_bgcolor_not_white = \
                self._figure.data[1].hoverlabel.bgcolor != "white"
            user_hoverlabel_bgcolor = False
            if plotly_params:
                if "node_trace" in plotly_params:
                    if "hoverlabel_bgcolor" in plotly_params["node_trace"]:
                        user_hoverlabel_bgcolor = True
            if is_bgcolor_not_white and not user_hoverlabel_bgcolor:
                colorscale_for_hoverlabel = \
                    self._figure.data[1].marker.colorscale

        observe_widgets(cover_params, cover_params_widgets)
        observe_widgets(cluster_params, cluster_params_widgets)
        observe_widgets(nerve_params, nerve_params_widgets)

        logs_box.observe(click_box, names="value")

        # Define containers for input widgets
        cover_title = HTML(value="<b>Cover parameters</b>")
        container_cover = widgets.VBox(
            children=[cover_title] + list(cover_params_widgets.values())
        )
        container_cover.layout.align_items = 'center'

        cluster_title = HTML(value="<b>Clusterer parameters</b>")
        container_cluster = widgets.VBox(
            children=[cluster_title] + list(cluster_params_widgets.values()),
        )
        container_cluster.layout.align_items = 'center'

        nerve_title = HTML(value="<b>Nerve parameters</b>")
        container_nerve = widgets.VBox(
            children=[nerve_title] + list(nerve_params_widgets.values()),
        )
        container_nerve.layout.align_items = 'center'

        container_parameters = widgets.HBox(
            children=[container_cover, container_cluster, container_nerve]
        )

        box = widgets.VBox([container_parameters, self._figure, valid,
                            logs_box, out])

        return box

    @property
    def graph_(self):
        return self._graph

    @property
    def pipeline_(self):
        return self._pipeline

    @property
    def color_features_(self):
        return self._color_data_transformed

    @property
    def node_summaries_(self):
        return self._node_colors_color_features

    @property
    def figure_(self):
        return self._figure
