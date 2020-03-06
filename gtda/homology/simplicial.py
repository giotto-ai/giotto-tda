"""Persistent homology on point clouds or finite metric spaces."""
# License: GNU AGPLv3

import numbers

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances

from ..base import PlotterMixin
from ._utils import _postprocess_diagrams
from ..externals.python import ripser, SparseRipsComplex, CechComplex
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import validate_params
from ..plots.homology import plot_diagram


@adapt_fit_transform_docs
class VietorisRipsPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """`Persistence diagrams <https://giotto.ai/theory>`_ resulting from
    `Vietoris-Rips filtrations <https://giotto.ai/theory>`_.

    Given a `point cloud <https://giotto.ai/theory>`_ in Euclidean space,
    or an abstract `metric space <https://giotto.ai/theory>`_ encoded by a
    distance matrix, information about the appearance and disappearance of
    topological features (technically, `homology classes
    <https://giotto.ai/theory>`_) of various dimensions and at different
    scales is summarised in the corresponding persistence diagram.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to `'precomputed'`, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and `metric` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan", or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    homology_dimensions : iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this
        death value is declared to be equal to `max_edge_length`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`.

    See also
    --------
    SparseRipsPersistence, ConsistentRescaling

    Notes
    -----
    `Ripser <https://github.com/Ripser/ripser>`_ is used as a C++ backend
    for computing Vietoris-Rips persistent homology. Python bindings were
    modified for performance from the `ripser.py
    <https://github.com/scikit-tda/ripser.py>`_ package.

    Persistence diagrams produced by this class must be interpreted with
    care due to the presence of padding triples which carry no information.
    See :meth:`transform` for additional information.

    References
    ----------
    [1] U. Bauer, "Ripser: efficient computation of Vietoris-Rips persistence \
        barcodes", 2019; `arXiv:1908.02518 \
        <https://arxiv.org/abs/1908.02518>`_.

    """

    _hyperparameters = {'max_edge_length': [numbers.Number, None],
                        'infinity_values_': [numbers.Number, None],
                        '_homology_dimensions': [list, [int, (0, np.inf)]],
                        'coeff': [int, (2, np.inf)]}

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=(0, 1), coeff=2, infinity_values=None,
                 n_jobs=None):
        self.metric = metric
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs

    def _ripser_diagram(self, X):
        Xdgms = ripser(X[X[:, 0] != np.inf],
                       maxdim=self._max_homology_dimension,
                       thresh=self.max_edge_length, coeff=self.coeff,
                       metric=self.metric)['dgms']

        if 0 in self._homology_dimensions:
            Xdgms[0] = Xdgms[0][:-1, :]  # Remove final death at np.inf

        # Add dimension as the third elements of each (b, d) tuple
        Xdgms = {dim: np.hstack([Xdgms[dim],
                                 dim * np.ones((Xdgms[dim].shape[0], 1),
                                               dtype=Xdgms[dim].dtype)])
                 for dim in self._homology_dimensions}
        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_points) or \
            (n_samples, n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            ``(n_points, n_points)``. Otherwise, each such entry will be
            interpreted as an ndarray of ``n_points`` row vectors in
            ``n_dimensions``-dimensional space.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)

        validate_params({**self.get_params(),
                         'infinity_values_': self.infinity_values_,
                         '_homology_dimensions': self._homology_dimensions},
                        self._hyperparameters)
        check_array(X, allow_nd=True, force_all_finite=False)

        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each point cloud or distance matrix in `X`, compute the
        relevant persistence diagram as an array of triples [b, d, q]. Each
        triple represents a persistent topological feature in dimension q
        (belonging to `homology_dimensions`) which is born at b and dies at d.
        Only triples in which b < d are meaningful. Triples in which b and d
        are equal ("diagonal elements") may be artificially introduced during
        the computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_points) or \
            (n_samples, n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            ``(n_points, n_points)``. Otherwise, each such entry will be
            interpreted as an ndarray of ``n_points`` row vectors in
            ``n_dimensions``-dimensional space.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_array(X, allow_nd=True, force_all_finite=False)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(self._ripser_diagram)(X[i])
                                          for i in range(len(X)))

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt

    def plot(self, X, sample=0, **layout):
        """Plot a single persistence diagram.

        Parameters
        ----------
        ----------
        X : ndarray of shape (n_samples, n_points, n_points) or \
            (n_samples, n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            ``(n_points, n_points)``. Otherwise, each such entry will be
            interpreted as an ndarray of ``n_points`` row vectors in
            ``n_dimensions``-dimensional space.

        sample : int, optional, default: ``0``
            Index of the sample to be plotted.

        layout : dict
            Dict of string/value properties that will be passed to the
            `plotly.graph_objs.Layout` constructor. Supported dict properties:

                angularaxis
                    plotly.graph_objects.layout.AngularAxis
                    instance or dict with compatible properties
                annotations
                    A tuple of
                    plotly.graph_objects.layout.Annotation
                    instances or dicts with compatible properties
                annotationdefaults
                    When used in a template (as
                    layout.template.layout.annotationdefaults),
                    sets the default property values to use for
                    elements of layout.annotations
                autosize
                    Determines whether or not a layout width or
                    height that has been left undefined by the user
                    is initialized on each relayout. Note that,
                    regardless of this attribute, an undefined
                    layout width or height is always initialized on
                    the first call to plot.
                bargap
                    Sets the gap (in plot fraction) between bars of
                    adjacent location coordinates.
                bargroupgap
                    Sets the gap (in plot fraction) between bars of
                    the same location coordinate.
                barmode
                    Determines how bars at the same location
                    coordinate are displayed on the graph. With
                    "stack", the bars are stacked on top of one
                    another With "relative", the bars are stacked
                    on top of one another, with negative values
                    below the axis, positive values above With
                    "group", the bars are plotted next to one
                    another centered around the shared location.
                    With "overlay", the bars are plotted over one
                    another, you might need to an "opacity" to see
                    multiple bars.
                barnorm
                    Sets the normalization for bar traces on the
                    graph. With "fraction", the value of each bar
                    is divided by the sum of all values at that
                    location coordinate. "percent" is the same but
                    multiplied by 100 to show percentages.
                boxgap
                    Sets the gap (in plot fraction) between boxes
                    of adjacent location coordinates. Has no effect
                    on traces that have "width" set.
                boxgroupgap
                    Sets the gap (in plot fraction) between boxes
                    of the same location coordinate. Has no effect
                    on traces that have "width" set.
                boxmode
                    Determines how boxes at the same location
                    coordinate are displayed on the graph. If
                    "group", the boxes are plotted next to one
                    another centered around the shared location. If
                    "overlay", the boxes are plotted over one
                    another, you might need to set "opacity" to see
                    them multiple boxes. Has no effect on traces
                    that have "width" set.
                calendar
                    Sets the default calendar system to use for
                    interpreting and displaying dates throughout
                    the plot.
                clickmode
                    Determines the mode of single click
                    interactions. "event" is the default value and
                    emits the `plotly_click` event. In addition
                    this mode emits the `plotly_selected` event in
                    drag modes "lasso" and "select", but with no
                    event data attached (kept for compatibility
                    reasons). The "select" flag enables selecting
                    single data points via click. This mode also
                    supports persistent selections, meaning that
                    pressing Shift while clicking, adds to /
                    subtracts from an existing selection. "select"
                    with `hovermode`: "x" can be confusing,
                    consider explicitly setting `hovermode`:
                    "closest" when using this feature. Selection
                    events are sent accordingly as long as "event"
                    flag is set as well. When the "event" flag is
                    missing, `plotly_click` and `plotly_selected`
                    events are not fired.
                coloraxis
                    plotly.graph_objects.layout.Coloraxis instance
                    or dict with compatible properties
                colorscale
                    plotly.graph_objects.layout.Colorscale instance
                    or dict with compatible properties
                colorway
                    Sets the default trace colors.
                datarevision
                    If provided, a changed value tells
                    `Plotly.react` that one or more data arrays has
                    changed. This way you can modify arrays in-
                    place rather than making a complete new copy
                    for an incremental change. If NOT provided,
                    `Plotly.react` assumes that data arrays are
                    being treated as immutable, thus any data array
                    with a different identity from its predecessor
                    contains new data.
                direction
                    Legacy polar charts are deprecated! Please
                    switch to "polar" subplots. Sets the direction
                    corresponding to positive angles in legacy
                    polar charts.
                dragmode
                    Determines the mode of drag interactions.
                    "select" and "lasso" apply only to scatter
                    traces with markers or text. "orbit" and
                    "turntable" apply only to 3D scenes.
                editrevision
                    Controls persistence of user-driven changes in
                    `editable: true` configuration, other than
                    trace names and axis titles. Defaults to
                    `layout.uirevision`.
                extendfunnelareacolors
                    If `true`, the funnelarea slice colors (whether
                    given by `funnelareacolorway` or inherited from
                    `colorway`) will be extended to three times its
                    original length by first repeating every color
                    20% lighter then each color 20% darker. This is
                    intended to reduce the likelihood of reusing
                    the same color when you have many slices, but
                    you can set `false` to disable. Colors provided
                    in the trace, using `marker.colors`, are never
                    extended.
                extendpiecolors
                    If `true`, the pie slice colors (whether given
                    by `piecolorway` or inherited from `colorway`)
                    will be extended to three times its original
                    length by first repeating every color 20%
                    lighter then each color 20% darker. This is
                    intended to reduce the likelihood of reusing
                    the same color when you have many slices, but
                    you can set `false` to disable. Colors provided
                    in the trace, using `marker.colors`, are never
                    extended.
                extendsunburstcolors
                    If `true`, the sunburst slice colors (whether
                    given by `sunburstcolorway` or inherited from
                    `colorway`) will be extended to three times its
                    original length by first repeating every color
                    20% lighter then each color 20% darker. This is
                    intended to reduce the likelihood of reusing
                    the same color when you have many slices, but
                    you can set `false` to disable. Colors provided
                    in the trace, using `marker.colors`, are never
                    extended.
                extendtreemapcolors
                    If `true`, the treemap slice colors (whether
                    given by `treemapcolorway` or inherited from
                    `colorway`) will be extended to three times its
                    original length by first repeating every color
                    20% lighter then each color 20% darker. This is
                    intended to reduce the likelihood of reusing
                    the same color when you have many slices, but
                    you can set `false` to disable. Colors provided
                    in the trace, using `marker.colors`, are never
                    extended.
                font
                    Sets the global font. Note that fonts used in
                    traces and other layout components inherit from
                    the global font.
                funnelareacolorway
                    Sets the default funnelarea slice colors.
                    Defaults to the main `colorway` used for trace
                    colors. If you specify a new list here it can
                    still be extended with lighter and darker
                    colors, see `extendfunnelareacolors`.
                funnelgap
                    Sets the gap (in plot fraction) between bars of
                    adjacent location coordinates.
                funnelgroupgap
                    Sets the gap (in plot fraction) between bars of
                    the same location coordinate.
                funnelmode
                    Determines how bars at the same location
                    coordinate are displayed on the graph. With
                    "stack", the bars are stacked on top of one
                    another With "group", the bars are plotted next
                    to one another centered around the shared
                    location. With "overlay", the bars are plotted
                    over one another, you might need to an
                    "opacity" to see multiple bars.
                geo
                    plotly.graph_objects.layout.Geo instance or
                    dict with compatible properties
                grid
                    plotly.graph_objects.layout.Grid instance or
                    dict with compatible properties
                height
                    Sets the plot's height (in px).
                hiddenlabels
                    hiddenlabels is the funnelarea & pie chart
                    analog of visible:'legendonly' but it can
                    contain many labels, and can simultaneously
                    hide slices from several pies/funnelarea charts
                hiddenlabelssrc
                    Sets the source reference on plot.ly for
                    hiddenlabels .
                hidesources
                    Determines whether or not a text link citing
                    the data source is placed at the bottom-right
                    cored of the figure. Has only an effect only on
                    graphs that have been generated via forked
                    graphs from the plotly service (at
                    https://plot.ly or on-premise).
                hoverdistance
                    Sets the default distance (in pixels) to look
                    for data to add hover labels (-1 means no
                    cutoff, 0 means no looking for data). This is
                    only a real distance for hovering on point-like
                    objects, like scatter points. For area-like
                    objects (bars, scatter fills, etc) hovering is
                    on inside the area and off outside, but these
                    objects will not supersede hover on point-like
                    objects in case of conflict.
                hoverlabel
                    plotly.graph_objects.layout.Hoverlabel instance
                    or dict with compatible properties
                hovermode
                    Determines the mode of hover interactions. If
                    `clickmode` includes the "select" flag,
                    `hovermode` defaults to "closest". If
                    `clickmode` lacks the "select" flag, it
                    defaults to "x" or "y" (depending on the
                    trace's `orientation` value) for plots based on
                    cartesian coordinates. For anything else the
                    default value is "closest".
                images
                    A tuple of plotly.graph_objects.layout.Image
                    instances or dicts with compatible properties
                imagedefaults
                    When used in a template (as
                    layout.template.layout.imagedefaults), sets the
                    default property values to use for elements of
                    layout.images
                legend
                    plotly.graph_objects.layout.Legend instance or
                    dict with compatible properties
                mapbox
                    plotly.graph_objects.layout.Mapbox instance or
                    dict with compatible properties
                margin
                    plotly.graph_objects.layout.Margin instance or
                    dict with compatible properties
                meta
                    Assigns extra meta information that can be used
                    in various `text` attributes. Attributes such
                    as the graph, axis and colorbar `title.text`,
                    annotation `text` `trace.name` in legend items,
                    `rangeselector`, `updatemenus` and `sliders`
                    `label` text all support `meta`. One can access
                    `meta` fields using template strings:
                    `%{meta[i]}` where `i` is the index of the
                    `meta` item in question. `meta` can also be an
                    object for example `{key: value}` which can be
                    accessed %{meta[key]}.
                metasrc
                    Sets the source reference on plot.ly for  meta
                    .
                modebar
                    plotly.graph_objects.layout.Modebar instance or
                    dict with compatible properties
                orientation
                    Legacy polar charts are deprecated! Please
                    switch to "polar" subplots. Rotates the entire
                    polar by the given angle in legacy polar
                    charts.
                paper_bgcolor
                    Sets the background color of the paper where
                    the graph is drawn.
                piecolorway
                    Sets the default pie slice colors. Defaults to
                    the main `colorway` used for trace colors. If
                    you specify a new list here it can still be
                    extended with lighter and darker colors, see
                    `extendpiecolors`.
                plot_bgcolor
                    Sets the background color of the plotting area
                    in-between x and y axes.
                polar
                    plotly.graph_objects.layout.Polar instance or
                    dict with compatible properties
                radialaxis
                    plotly.graph_objects.layout.RadialAxis instance
                    or dict with compatible properties
                scene
                    plotly.graph_objects.layout.Scene instance or
                    dict with compatible properties
                selectdirection
                    When "dragmode" is set to "select", this limits
                    the selection of the drag to horizontal,
                    vertical or diagonal. "h" only allows
                    horizontal selection, "v" only vertical, "d"
                    only diagonal and "any" sets no limit.
                selectionrevision
                    Controls persistence of user-driven changes in
                    selected points from all traces.
                separators
                    Sets the decimal and thousand separators. For
                    example, *. * puts a '.' before decimals and a
                    space between thousands. In English locales,
                    dflt is ".," but other locales may alter this
                    default.
                shapes
                    A tuple of plotly.graph_objects.layout.Shape
                    instances or dicts with compatible properties
                shapedefaults
                    When used in a template (as
                    layout.template.layout.shapedefaults), sets the
                    default property values to use for elements of
                    layout.shapes
                showlegend
                    Determines whether or not a legend is drawn.
                    Default is `true` if there is a trace to show
                    and any of these: a) Two or more traces would
                    by default be shown in the legend. b) One pie
                    trace is shown in the legend. c) One trace is
                    explicitly given with `showlegend: true`.
                sliders
                    A tuple of plotly.graph_objects.layout.Slider
                    instances or dicts with compatible properties
                sliderdefaults
                    When used in a template (as
                    layout.template.layout.sliderdefaults), sets
                    the default property values to use for elements
                    of layout.sliders
                spikedistance
                    Sets the default distance (in pixels) to look
                    for data to draw spikelines to (-1 means no
                    cutoff, 0 means no looking for data). As with
                    hoverdistance, distance does not apply to area-
                    like objects. In addition, some objects can be
                    hovered on but will not generate spikelines,
                    such as scatter fills.
                sunburstcolorway
                    Sets the default sunburst slice colors.
                    Defaults to the main `colorway` used for trace
                    colors. If you specify a new list here it can
                    still be extended with lighter and darker
                    colors, see `extendsunburstcolors`.
                template
                    Default attributes to be applied to the plot.
                    This should be a dict with format: `{'layout':
                    layoutTemplate, 'data': {trace_type:
                    [traceTemplate, ...], ...}}` where
                    `layoutTemplate` is a dict matching the
                    structure of `figure.layout` and
                    `traceTemplate` is a dict matching the
                    structure of the trace with type `trace_type`
                    (e.g. 'scatter'). Alternatively, this may be
                    specified as an instance of
                    plotly.graph_objs.layout.Template.  Trace
                    templates are applied cyclically to traces of
                    each type. Container arrays (eg `annotations`)
                    have special handling: An object ending in
                    `defaults` (eg `annotationdefaults`) is applied
                    to each array item. But if an item has a
                    `templateitemname` key we look in the template
                    array for an item with matching `name` and
                    apply that instead. If no matching `name` is
                    found we mark the item invisible. Any named
                    template item not referenced is appended to the
                    end of the array, so this can be used to add a
                    watermark annotation or a logo image, for
                    example. To omit one of these items on the
                    plot, make an item with matching
                    `templateitemname` and `visible: false`.
                ternary
                    plotly.graph_objects.layout.Ternary instance or
                    dict with compatible properties
                title
                    plotly.graph_objects.layout.Title instance or
                    dict with compatible properties
                titlefont
                    Deprecated: Please use layout.title.font
                    instead. Sets the title font. Note that the
                    title's font used to be customized by the now
                    deprecated `titlefont` attribute.
                transition
                    Sets transition options used during
                    Plotly.react updates.
                treemapcolorway
                    Sets the default treemap slice colors. Defaults
                    to the main `colorway` used for trace colors.
                    If you specify a new list here it can still be
                    extended with lighter and darker colors, see
                    `extendtreemapcolors`.
                uirevision
                    Used to allow user interactions with the plot
                    to persist after `Plotly.react` calls that are
                    unaware of these interactions. If `uirevision`
                    is omitted, or if it is given and it changed
                    from the previous `Plotly.react` call, the
                    exact new figure is used. If `uirevision` is
                    truthy and did NOT change, any attribute that
                    has been affected by user interactions and did
                    not receive a different value in the new figure
                    will keep the interaction value.
                    `layout.uirevision` attribute serves as the
                    default for `uirevision` attributes in various
                    sub-containers. For finer control you can set
                    these sub-attributes directly. For example, if
                    your app separately controls the data on the x
                    and y axes you might set
                    `xaxis.uirevision=*time*` and
                    `yaxis.uirevision=*cost*`. Then if only the y
                    data is changed, you can update
                    `yaxis.uirevision=*quantity*` and the y axis
                    range will reset but the x axis range will
                    retain any user-driven zoom.
                uniformtext
                    plotly.graph_objects.layout.Uniformtext
                    instance or dict with compatible properties
                updatemenus
                    A tuple of
                    plotly.graph_objects.layout.Updatemenu
                    instances or dicts with compatible properties
                updatemenudefaults
                    When used in a template (as
                    layout.template.layout.updatemenudefaults),
                    sets the default property values to use for
                    elements of layout.updatemenus
                violingap
                    Sets the gap (in plot fraction) between violins
                    of adjacent location coordinates. Has no effect
                    on traces that have "width" set.
                violingroupgap
                    Sets the gap (in plot fraction) between violins
                    of the same location coordinate. Has no effect
                    on traces that have "width" set.
                violinmode
                    Determines how violins at the same location
                    coordinate are displayed on the graph. If
                    "group", the violins are plotted next to one
                    another centered around the shared location. If
                    "overlay", the violins are plotted over one
                    another, you might need to set "opacity" to see
                    them multiple violins. Has no effect on traces
                    that have "width" set.
                waterfallgap
                    Sets the gap (in plot fraction) between bars of
                    adjacent location coordinates.
                waterfallgroupgap
                    Sets the gap (in plot fraction) between bars of
                    the same location coordinate.
                waterfallmode
                    Determines how bars at the same location
                    coordinate are displayed on the graph. With
                    "group", the bars are plotted next to one
                    another centered around the shared location.
                    With "overlay", the bars are plotted over one
                    another, you might need to an "opacity" to see
                    multiple bars.
                width
                    Sets the plot's width (in px).
                xaxis
                    plotly.graph_objects.layout.XAxis instance or
                    dict with compatible properties
                yaxis
                    plotly.graph_objects.layout.YAxis instance or
                    dict with compatible properties

        """
        return plot_diagram(X[sample],
                            homology_dimensions=self.homology_dimensions,
                            **layout)



@adapt_fit_transform_docs
class SparseRipsPersistence(BaseEstimator, TransformerMixin):
    """`Persistence diagrams <https://giotto.ai/theory>`_ resulting from
    `Sparse Vietoris-Rips filtrations <https://giotto.ai/theory>`_.

    Given a `point cloud <https://giotto.ai/theory>`_ in Euclidean space,
    or an abstract `metric space <https://giotto.ai/theory>`_ encoded by a
    distance matrix, information about the appearance and disappearance of
    topological features (technically, `homology classes
    <https://giotto.ai/theory>`_) of various dimensions and at different
    scales is summarised in the corresponding persistence diagram.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to `'precomputed'`, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and `metric` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan", or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    homology_dimensions : iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    epsilon : float between 0. and 1., optional, default: ``0.1``
        Parameter controlling the approximation to the exact Vietoris-Rips
        filtration. If set to `0.`, :class:`SparseRipsPersistence` leads to
        the same results as :class:`VietorisRipsPersistence` but is slower.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    infinity_values : float or None, default : ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this
        death value is declared to be equal to `max_edge_length`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`. Set in :meth:`fit`.

    See also
    --------
    VietorisRipsPersistence, ConsistentRescaling

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing sparse Vietoris-Rips persistent homology. Python bindings
    were modified for performance.

    Persistence diagrams produced by this class must be interpreted with
    care due to the presence of padding triples which carry no information.
    See :meth:`transform` for additional information.

    References
    ----------
    [1] C. Maria, "Persistent Cohomology", 2020; `GUDHI User and Reference \
        Manual <http://gudhi.gforge.inria.fr/doc/3.1.0/group__persistent_\
        cohomology.html>`_.

    """
    _hyperparameters = {'epsilon': [numbers.Number, (0., 1.)],
                        'max_edge_length': [numbers.Number, None],
                        'infinity_values_': [numbers.Number, None],
                        '_homology_dimensions': [list, [int, (0, np.inf)]],
                        'coeff': [int, (2, np.inf)]}

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=(0, 1), coeff=2, epsilon=0.1,
                 infinity_values=None, n_jobs=None):
        self.metric = metric
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.epsilon = epsilon
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs

    def _gudhi_diagram(self, X):
        Xdgms = pairwise_distances(X, metric=self.metric)
        sparse_rips_complex = SparseRipsComplex(
            distance_matrix=Xdgms, max_edge_length=self.max_edge_length,
            sparse=self.epsilon)
        simplex_tree = sparse_rips_complex.create_simplex_tree(
            max_dimension=max(self._homology_dimensions) + 1)
        Xdgms = simplex_tree.persistence(
            homology_coeff_field=self.coeff, min_persistence=0)

        # Separate diagrams by homology dimensions
        Xdgms = {dim: np.array([Xdgms[i][1] for i in range(len(Xdgms))
                                if Xdgms[i][0] == dim]).reshape((-1, 2))
                 for dim in self.homology_dimensions}

        if 0 in self._homology_dimensions:
            Xdgms[0] = Xdgms[0][1:, :]  # Remove final death at np.inf

        # Add dimension as the third elements of each (b, d) tuple
        Xdgms = {dim: np.hstack([Xdgms[dim],
                                 dim * np.ones((Xdgms[dim].shape[0], 1),
                                               dtype=Xdgms[dim].dtype)])
                 for dim in self._homology_dimensions}
        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_points) or \
            (n_samples, n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            ``(n_points, n_points)``. Otherwise, each such entry will be
            interpreted as an ndarray of ``n_points`` row vectors in
            ``n_dimensions``-dimensional space.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)

        validate_params({**self.get_params(),
                         'infinity_values_': self.infinity_values_,
                         '_homology_dimensions': self._homology_dimensions},
                        self._hyperparameters)
        check_array(X, allow_nd=True, force_all_finite=False)

        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each point cloud or distance matrix in `X`, compute the
        relevant persistence diagram as an array of triples [b, d, q]. Each
        triple represents a persistent topological feature in dimension q
        (belonging to `homology_dimensions`) which is born at b and dies at d.
        Only triples in which b < d are meaningful. Triples in which b and d
        are equal ("diagonal elements") may be artificially introduced during
        the computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_points) or \
            (n_samples, n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            ``(n_points, n_points)``. Otherwise, each such entry will be
            interpreted as an ndarray of ``n_points`` row vectors in
            ``n_dimensions``-dimensional space.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_array(X, allow_nd=True, force_all_finite=False)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._gudhi_diagram)(X[i, :, :]) for i in range(
                X.shape[0]))

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt


@adapt_fit_transform_docs
class EuclideanCechPersistence(BaseEstimator, TransformerMixin):
    """`Persistence diagrams <https://giotto.ai/theory>`_ resulting from
    `Cech filtrations <https://giotto.ai/theory>`_.

    Given a `point cloud <https://giotto.ai/theory>`_ in Euclidean space,
    information about the appearance and disappearance of topological
    features (technically, `homology classes <https://giotto.ai/theory>`_) of
    various dimensions and at different scales is summarised in the
    corresponding persistence diagram.

    Parameters
    ----------
    homology_dimensions : iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`.

    See also
    --------
    VietorisRipsPersistence, SparseRipsPersistence, ConsistentRescaling

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing Cech persistent homology. Python bindings were modified
    for performance.

    Persistence diagrams produced by this class must be interpreted with
    care due to the presence of padding triples which carry no information.
    See :meth:`transform` for additional information.

    References
    ----------
    [1] C. Maria, "Persistent Cohomology", 2020; `GUDHI User and Reference \
        Manual <http://gudhi.gforge.inria.fr/doc/3.1.0/group__persistent_\
        cohomology.html>`_.

    """
    _hyperparameters = {'max_edge_length': [numbers.Number, None],
                        'infinity_values_': [numbers.Number, None],
                        '_homology_dimensions': [list, [int, (0, np.inf)]],
                        'coeff': [int, (2, np.inf)]}

    def __init__(self, max_edge_length=np.inf, homology_dimensions=(0, 1),
                 coeff=2, infinity_values=None, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs

    def _gudhi_diagram(self, X):
        cech_complex = CechComplex(points=X, max_radius=self.max_edge_length)
        simplex_tree = cech_complex.create_simplex_tree(
            max_dimension=max(self._homology_dimensions) + 1)
        Xdgms = simplex_tree.persistence(
            homology_coeff_field=self.coeff, min_persistence=0)

        # Separate diagrams by homology dimensions
        Xdgms = {dim: np.array([Xdgms[i][1] for i in range(len(Xdgms))
                                if Xdgms[i][0] == dim]).reshape((-1, 2))
                 for dim in self.homology_dimensions}

        if 0 in self._homology_dimensions:
            Xdgms[0] = Xdgms[0][1:, :]  # Remove final death at np.inf

        # Add dimension as the third elements of each (b, d) tuple
        Xdgms = {dim: np.hstack([Xdgms[dim],
                                 dim * np.ones((Xdgms[dim].shape[0], 1),
                                               dtype=Xdgms[dim].dtype)])
                 for dim in self._homology_dimensions}
        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data. Each entry along axis 0 is a point cloud of
            ``n_points`` row vectors in ``n_dimensions``-dimensional space.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)

        validate_params({**self.get_params(),
                         'infinity_values_': self.infinity_values_,
                         '_homology_dimensions': self._homology_dimensions},
                        self._hyperparameters)
        check_array(X, allow_nd=True)

        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each point cloud in `X`, compute the relevant persistence
        diagram as an array of triples [b, d, q]. Each triple represents a
        persistent topological feature in dimension q (belonging to
        `homology_dimensions`) which is born at b and dies at d. Only triples
        in which b < d are meaningful. Triples in which b and d are equal
        ("diagonal elements") may be artificially introduced during the
        computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data. Each entry along axis 0 is a point cloud of
            ``n_points`` row vectors in ``n_dimensions``-dimensional space.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays in
            `X`. ``n_features`` equals :math:`\\sum_q n_q`, where :math:`n_q`
            is the maximum number of topological features in dimension
            :math:`q` across all samples in `X`.

        """
        check_is_fitted(self)
        X = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._gudhi_diagram)(X[i, :, :]) for i in range(
                X.shape[0]))

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt
