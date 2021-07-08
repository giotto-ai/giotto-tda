"""Graph layout functions and plotly layout functions."""
# License: GNU AGPLv3

from copy import deepcopy
from functools import reduce, partial
from operator import iconcat
from warnings import warn

import numpy as np
import plotly.graph_objs as go
from sklearn.utils.validation import check_array

PLOT_OPTIONS_NODE_TRACE_DEFAULTS = {
    "name": "node_trace",
    "mode": "markers",
    "hoverinfo": "text",
    "marker": {
        "colorscale": "viridis",
        "opacity": 1.,
        "showscale": True,
        "reversescale": False,
        "line": {"width": 1, "color": "#888"},
        "sizemode": "area",
        "sizemin": 4,
        "colorbar": {
            "thickness": 15, "title": "", "xanchor": "left",
            "titleside": "right"
            }
        }
    }

PLOT_OPTIONS_EDGE_TRACE_DEFAULTS = {
    "name": "edge_trace",
    "mode": "lines",
    "line": {"width": 1, "color": "#888"},
    "hoverinfo": "none"
    }

PLOT_OPTIONS_LAYOUT_COMMON_DEFAULTS = {
    "showlegend": False,
    "hovermode": "closest",
    "title": "",
    "margin": {"b": 20, "l": 5, "r": 5, "t": 40},
    "autosize": False,
    "annotations": []
    }

PLOT_OPTIONS_LAYOUT_AXES_DEFAULTS_2D = {
    "title": "", "showgrid": False, "zeroline": False, "showticklabels": False,
    "ticks": "", "showline": False
    }

PLOT_OPTIONS_LAYOUT_AXES_DEFAULTS_3D = {
    "title": "", "showbackground": False, "showline": False, "zeroline": False,
    "showgrid": False, "showticklabels": False,
    }

PLOT_OPTIONS_LAYOUT_DEFAULTS = {
    "common": PLOT_OPTIONS_LAYOUT_COMMON_DEFAULTS,
    2: {
        "template": "simple_white",
        "xaxis": PLOT_OPTIONS_LAYOUT_AXES_DEFAULTS_2D,
        "yaxis": PLOT_OPTIONS_LAYOUT_AXES_DEFAULTS_2D
        },
    3: {
        "scene": {
            "xaxis": PLOT_OPTIONS_LAYOUT_AXES_DEFAULTS_3D,
            "yaxis": PLOT_OPTIONS_LAYOUT_AXES_DEFAULTS_3D,
            "zaxis": PLOT_OPTIONS_LAYOUT_AXES_DEFAULTS_3D
            }
        }
    }


def _set_node_sizeref(node_sizes, node_scale=12):
    # Formula from Plotly https://plot.ly/python/bubble-charts/
    return 2. * max(node_sizes) / (node_scale ** 2)


def _round_to_n_sig_figs(x, n=3):
    """Round a number x to n significant figures."""
    if n is None:
        return x
    if not x:
        return 0
    return np.round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))


def _get_node_size(node_elements):
    # TODO: Add doc strings to all functions
    return list(map(len, node_elements))


def _get_node_text(
        node_ids, pullback_set_labels, partial_cluster_labels,
        num_node_elements, node_summary_statistics
        ):
    return [
        f"Node ID: {node_id}<br>Pullback set label: {pullback_set_label}<br>"
        f"Partial cluster label: {partial_cluster_label}<br>Node size: "
        f"{num_elements}<br>Summary statistic: {node_summary_statistic}"
        for (node_id, pullback_set_label, partial_cluster_label, num_elements,
             node_summary_statistic)
        in zip(node_ids, pullback_set_labels, partial_cluster_labels,
               num_node_elements, node_summary_statistics)
        ]


def _get_node_statistics(color_data_transformed, node_elements,
                         node_color_statistic):
    n_columns = color_data_transformed.shape[1]

    return np.array([[node_color_statistic(color_data_transformed[itr, i])
                      for i in range(n_columns)]
                     for itr in node_elements])


def _get_column_color_buttons(
        node_colors_color_features, hovertext_color_features,
        colorscale_for_hoverlabel, n_sig_figs, column_names_dropdown
        ):
    # TODO: Consider opting for just-in-time computation instead of computing
    # all node summary values ahead of time. Solution should preserve scroll
    # zoom functionality of 2D static visualisation.
    def replace_summary_statistic(current_hovertext, new_statistic):
        pos = current_hovertext.rfind(" ")
        new_hovertext = current_hovertext[:pos] + \
            f" {_round_to_n_sig_figs(new_statistic, n=n_sig_figs)}"
        return new_hovertext

    column_color_buttons = [
        {
            "args": [{
                "marker.color": [None, node_colors_color_features[:, 0]],
                "hovertext": [None, hovertext_color_features]
                }],
            "label": f"{column_names_dropdown[0]}",
            "method": "restyle"
            }
        ]

    for column in range(1, len(column_names_dropdown)):
        node_colors = node_colors_color_features[:, column]
        hovertext = list(map(replace_summary_statistic,
                             hovertext_color_features, node_colors))

        new_button = {
            "args": [{
                "marker.color": [None, node_colors],
                "hovertext": [None, hovertext]
                }],
            "label": f"{column_names_dropdown[column]}",
            "method": "restyle"
            }

        if colorscale_for_hoverlabel is not None:
            min_col = np.min(node_colors)
            max_col = np.max(node_colors)
            new_button["args"][0]["hoverlabel.bgcolor"] = [
                None,
                _get_colors_for_vals(node_colors, min_col, max_col,
                                     colorscale_for_hoverlabel)
                ]

        column_color_buttons.append(new_button)

    return column_color_buttons


def _infer_color_features_kind(color_features):
    """Determine whether color_features is array, pandas dataframe, callable,
    or scikit-learn (fit-)transformer."""
    if hasattr(color_features, "dtype") or hasattr(color_features, "dtypes"):
        raise ValueError("`color_features` should not be a numpy array or "
                         "pandas dataframe.")
    elif hasattr(color_features, "fit_transform"):
        color_features_kind = "fit_transformer"
    elif hasattr(color_features, "transform"):
        color_features_kind = "transformer"
    elif callable(color_features):
        color_features_kind = "callable"
    elif color_features is None:
        color_features_kind = "none"
    else:  # Assume color_features is a selection of columns
        color_features_kind = "else"

    return color_features_kind


def _validate_color_kwargs(graph, data, color_data, color_features,
                           node_color_statistic, interactive=False):
    if node_color_statistic is None:
        node_color_statistic = np.mean

    if color_data is None:
        color_data = np.arange(len(data))

    color_data_checked = check_array(color_data, ensure_2d=False, dtype=None)

    # Simple duck typing to determine whether `color_data` is likely a pandas
    # dataframe or pandas series
    is_color_data_dataframe = hasattr(color_data, "columns")
    is_color_data_series = hasattr(color_data, "name")

    if len(color_data) != len(data):
        raise ValueError("`color_data` and `data` must have the same length.")
    if not (is_color_data_dataframe or is_color_data_series):
        color_data = color_data_checked

    # Determine whether node_color_statistic is an array of node colors
    is_node_color_statistic_ndarray = hasattr(node_color_statistic, "dtype")
    is_node_color_statistic_callable = callable(node_color_statistic)

    column_names_dropdown = None
    if interactive:
        if not is_node_color_statistic_callable:
            raise ValueError("`node_color_statistic` must be a callable for "
                             "interactive plots.")
    else:
        node_elements = graph.vs["node_elements"]
        if is_node_color_statistic_ndarray:
            node_color_statistic = check_array(node_color_statistic,
                                               ensure_2d=False)
            len_colors = len(node_color_statistic)
            n_nodes = len(node_elements)
            if len_colors != n_nodes:
                raise ValueError(f"`node_color_statistic` must have as many "
                                 f"entries as there are nodes in the Mapper "
                                 f"graph. {len_colors} != {n_nodes} detected.")
            if len(node_color_statistic.shape) > 1:
                column_names_dropdown = range(node_color_statistic.shape[1])
            else:
                node_color_statistic = \
                    node_color_statistic.reshape((len_colors, -1))
        elif not is_node_color_statistic_callable:
            raise ValueError("`node_color_statistic` must be a callable or "
                             "an ndarray for static plots.")

    color_features_kind = _infer_color_features_kind(color_features)
    if color_features_kind == "fit_transformer":
        color_data_transformed = color_features.fit_transform(color_data)
    elif color_features_kind == "transformer":
        color_data_transformed = color_features.transform(color_data)
    elif color_features_kind == "callable":
        color_data_transformed = color_features(color_data)
        # If outcome is a pandas dataframe, save column names before converting
        # to numpy array. If a pandas series, just convert to numpy
        if hasattr(color_data_transformed, "columns"):
            column_names_dropdown = color_data_transformed.columns
            color_data_transformed = color_data_transformed.to_numpy()
        elif hasattr(color_data_transformed, "name"):
            color_data_transformed = color_data_transformed.to_numpy()
    elif color_features_kind == "none":
        if not (is_color_data_dataframe or is_color_data_series):
            color_data_transformed = color_data
        else:  # Dataframe or series
            if is_color_data_dataframe:
                column_names_dropdown = color_data.columns
            color_data_transformed = color_data.to_numpy()
    else:  # Case of column or sequence of columns
        if is_color_data_dataframe:
            color_data_transformed = color_data[color_features]
            column_names_dropdown = getattr(color_data_transformed, "columns",
                                            None)
            color_data_transformed = color_data_transformed.to_numpy()
        elif is_color_data_series:
            raise ValueError(
                "If `color_data` is a pandas series, `color_features` can "
                "only be a scikit-learn transformer, a callable, or None."
                )
        else:
            color_data_transformed = color_data[:, color_features]
            if len(color_data_transformed.shape) > 1:
                column_names_dropdown = color_features

    if len(color_data_transformed.shape) > 1:
        if column_names_dropdown is None:
            column_names_dropdown = range(color_data_transformed.shape[1])

    color_data_transformed = \
        color_data_transformed.reshape((len(color_data_transformed), -1))

    return (color_data_transformed, column_names_dropdown,
            node_color_statistic)


def _calculate_graph_data(
        graph, color_data_transformed, node_color_statistic, layout,
        layout_dim, n_sig_figs, node_scale
        ):
    node_elements = graph.vs["node_elements"]

    if not hasattr(node_color_statistic, "dtype"):
        node_colors_color_features = \
            _get_node_statistics(color_data_transformed, node_elements,
                                 node_color_statistic)
    else:
        node_colors_color_features = node_color_statistic

    # Load defaults for node and edge traces
    plot_options = {
        "node_trace": deepcopy(PLOT_OPTIONS_NODE_TRACE_DEFAULTS),
        "edge_trace": deepcopy(PLOT_OPTIONS_EDGE_TRACE_DEFAULTS)
        }

    # Update size and color of nodes with zeroth column of
    # `node_colors_color_features`
    node_sizes = _get_node_size(node_elements)
    plot_options["node_trace"]["marker"].update({
        "size": node_sizes,
        "sizeref": _set_node_sizeref(node_sizes, node_scale=node_scale),
        "color": node_colors_color_features[:, 0]
        })

    # Generate hovertext
    node_ids = graph.vs.indices
    pullback_set_ids = graph.vs["pullback_set_label"]
    partial_cluster_labels = graph.vs["partial_cluster_label"]
    num_node_elements = map(len, node_elements)
    node_colors_round = map(partial(_round_to_n_sig_figs, n=n_sig_figs),
                            node_colors_color_features[:, 0])
    plot_options["node_trace"]["hovertext"] = _get_node_text(
        node_ids, pullback_set_ids, partial_cluster_labels,
        num_node_elements, node_colors_round
        )

    # Compute graph layout
    if layout_dim not in [2, 3]:
        raise ValueError(
            f"`layout_dim` must be either 2 or 3. {layout_dim} entered."
            )
    node_pos = np.asarray(graph.layout(layout, dim=layout_dim).coords)

    # Store x and y coordinates of edge endpoints
    edge_x = list(
        reduce(
            iconcat, map(
                lambda e: [node_pos[e.source, 0], node_pos[e.target, 0],
                           None],
                graph.es
                ), []
            )
        )
    edge_y = list(
        reduce(
            iconcat, map(
                lambda e: [node_pos[e.source, 1], node_pos[e.target, 1],
                           None],
                graph.es
                ), []
            )
        )

    if layout_dim == 2:
        node_trace = go.Scatter(
            x=node_pos[:, 0], y=node_pos[:, 1], **plot_options["node_trace"]
            )

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, **plot_options["edge_trace"]
            )

    else:
        node_trace = go.Scatter3d(
            x=node_pos[:, 0], y=node_pos[:, 1], z=node_pos[:, 2],
            **plot_options["node_trace"]
            )

        edge_z = list(
            reduce(
                iconcat, map(
                    lambda e: [node_pos[e.source][2], node_pos[e.target][2],
                               None],
                    graph.es
                    ), []
                )
            )
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, **plot_options["edge_trace"]
            )

    return edge_trace, node_trace, node_colors_color_features


def _produce_static_figure(edge_trace, node_trace, node_colors_color_features,
                           column_names_dropdown, layout_dim, n_sig_figs,
                           plotly_params):
    # Define layout options
    layout_options = go.Layout(
        **PLOT_OPTIONS_LAYOUT_DEFAULTS["common"],
        **PLOT_OPTIONS_LAYOUT_DEFAULTS[layout_dim]
        )

    fig = go.FigureWidget(data=[edge_trace, node_trace], layout=layout_options)

    _plotly_params = deepcopy(plotly_params)

    # When laying out the graph in 3D, plotly does not automatically give
    # the background hoverlabel the same color as the respective marker,
    # so we do this by hand here.
    # TODO: Extract logic so as to avoid repetitions in interactive version
    colorscale_for_hoverlabel = None
    if layout_dim == 3:
        compute_hoverlabel_bgcolor = True
        if _plotly_params:
            if "node_trace" in _plotly_params:
                if "hoverlabel_bgcolor" in _plotly_params["node_trace"]:
                    fig.update_traces(
                        hoverlabel_bgcolor=_plotly_params["node_trace"].pop(
                            "hoverlabel_bgcolor"
                            ),
                        selector={"name": "node_trace"}
                        )
                    compute_hoverlabel_bgcolor = False
                if "marker_colorscale" in _plotly_params["node_trace"]:
                    fig.update_traces(
                        marker_colorscale=_plotly_params["node_trace"].pop(
                            "marker_colorscale"
                            ),
                        selector={"name": "node_trace"}
                        )

        if compute_hoverlabel_bgcolor:
            colorscale_for_hoverlabel = fig.data[1].marker.colorscale
            min_col = np.min(node_colors_color_features[:, 0])
            max_col = np.max(node_colors_color_features[:, 0])
            try:
                hoverlabel_bgcolor = _get_colors_for_vals(
                    node_colors_color_features[:, 0], min_col, max_col,
                    colorscale_for_hoverlabel
                    )
            except Exception as e:
                if e.args[0] == "This colorscale is not supported.":
                    warn("Data-dependent background hoverlabel colors cannot "
                         "be generated with this choice of colorscale. Please "
                         "use a standard hex- or RGB-formatted colorscale.",
                         RuntimeWarning)
                else:
                    warn("Something went wrong in generating data-dependent "
                         "background hoverlabel colors. All background "
                         "hoverlabel colors will be set to white.",
                         RuntimeWarning)
                hoverlabel_bgcolor = "white"
                colorscale_for_hoverlabel = None
            fig.update_traces(
                hoverlabel_bgcolor=hoverlabel_bgcolor,
                selector={"name": "node_trace"}
                )

    # Produce dropdown menu if `node_colors_color_features` has more than
    # one column
    if node_colors_color_features.shape[1] > 1:
        hovertext_color_features = node_trace.hovertext
        column_color_buttons = _get_column_color_buttons(
            node_colors_color_features, hovertext_color_features,
            colorscale_for_hoverlabel, n_sig_figs, column_names_dropdown
            )

        button_height = 1.1
        fig.update_layout(
            updatemenus=[
                go.layout.Updatemenu(buttons=column_color_buttons,
                                     direction="down",
                                     pad={"r": 10, "t": 10},
                                     showactive=True,
                                     x=0.11,
                                     xanchor="left",
                                     y=button_height,
                                     yanchor="top")
                ]
            )

        fig.add_annotation(
            go.layout.Annotation(text="Color by:",
                                 x=0,
                                 xref="paper",
                                 y=button_height - 0.045,
                                 yref="paper",
                                 align="left",
                                 showarrow=False)
            )

    # Update traces and layout according to user input
    if _plotly_params:
        for key in ["node_trace", "edge_trace"]:
            fig.update_traces(
                _plotly_params.pop(key, None),
                selector={"name": key}
                )
        fig.update_layout(_plotly_params.pop("layout", None))

    return fig


def _hex_to_rgb(value):
    """Convert a hex-formatted color to rgb, ignoring alpha values."""
    value = value.lstrip("#")
    return [int(value[i:i + 2], 16) for i in range(0, 6, 2)]


def _rbg_to_hex(c):
    """Convert an rgb-formatted color to hex, ignoring alpha values."""
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def _get_colors_for_vals(vals, vmin, vmax, colorscale, return_hex=True):
    """Given a float array vals, interpolate based on a colorscale to obtain
    rgb or hex colors. Inspired by
    `user empet's answer in \
    <community.plotly.com/t/hover-background-color-on-scatter-3d/9185/6>`_."""
    from numbers import Number
    from ast import literal_eval

    if vmin >= vmax:
        raise ValueError("`vmin` should be < `vmax`.")

    if (len(colorscale[0]) == 2) and isinstance(colorscale[0][0], Number):
        scale, colors = zip(*colorscale)
    else:
        scale = np.linspace(0, 1, num=len(colorscale))
        colors = colorscale
    scale = np.asarray(scale)

    if colors[0][:3] == "rgb":
        colors = np.asarray([literal_eval(color[3:]) for color in colors],
                            dtype=float)
    elif colors[0][0] == "#":
        colors = np.asarray(list(map(_hex_to_rgb, colors)), dtype=float)
    else:
        raise ValueError("This colorscale is not supported.")

    colorscale = np.hstack([scale.reshape(-1, 1), colors])
    colorscale = np.vstack([colorscale, colorscale[0, :]])
    colorscale_diffs = np.diff(colorscale, axis=0)
    colorscale_diff_ratios = colorscale_diffs[:, 1:] / colorscale_diffs[:, [0]]
    colorscale_diff_ratios[-1, :] = np.zeros(3)

    vals_scaled = (vals - vmin) / (vmax - vmin)

    left_bin_indices = np.digitize(vals_scaled, scale) - 1
    left_endpts = colorscale[left_bin_indices]
    vals_scaled -= left_endpts[:, 0]
    diff_ratios = colorscale_diff_ratios[left_bin_indices]

    vals_rgb = (
            left_endpts[:, 1:] + diff_ratios * vals_scaled[:, np.newaxis] + 0.5
        ).astype(np.uint8)

    if return_hex:
        return list(map(_rbg_to_hex, vals_rgb))

    return [f"rgb{tuple(v)}" for v in vals_rgb]
