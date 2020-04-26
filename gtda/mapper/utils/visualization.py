"""Graph layout functions and plotly layout functions."""
# License: GNU AGPLv3

import operator
from copy import deepcopy
from functools import reduce, partial

import numpy as np
import plotly.graph_objs as go
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

PLOT_OPTIONS_NODE_TRACE_DEFAULTS = {
    "mode": "markers",
    "hoverinfo": "text",
    "marker": {
        "colorscale": "viridis",
        "showscale": True,
        "reversescale": False,
        "line": {"width": .5, "color": "#888"},
        "sizemode": "area",
        "sizemin": 4,
        "colorbar": {
            "thickness": 15, "title": "", "xanchor": "left",
            "titleside": "right"
        },
        "line_width": 2
    }
}

PLOT_OPTIONS_EDGE_TRACE_DEFAULTS = {
    "mode": "lines",
    "line": {"color": "#888", "width": 1},
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


def set_node_sizeref(node_elements, node_scale=12):
    # Formula from Plotly https://plot.ly/python/bubble-charts/
    return 2. * max(_get_node_size(node_elements)) / (node_scale ** 2)


def _get_node_size(node_elements):
    # TODO: Add doc strings to all functions
    return list(map(len, node_elements))


def _get_node_text(graph):
    return [
        f"Node ID: {node_id}<br>Node size: {len(node_elements)}"
        f"<br>Summary statistic: {node_summary_statistic}"
        for node_id, node_elements, node_summary_statistic in zip(
            graph["node_metadata"]["node_id"],
            graph["node_metadata"]["node_elements"],
            graph["node_metadata"]["node_summary_statistic"],
            )
        ]


def _get_node_summary(data, node_elements, summary_statistic=np.mean):
    return np.asarray(
        list(map(summary_statistic, [data[itr] for itr in node_elements]))
    )


def _get_column_color_buttons(
        data, is_data_dataframe, node_elements, node_colors_color_variable,
        color_variable_min, color_variable_max, summary_statistic, colorscale
):
    # TODO: Consider opting for just-in-time computation instead of computing
    # all node summary values ahead of time. Solution should preserve scroll
    # zoom functionality of 2D static visualisation.
    if is_data_dataframe:
        columns_to_color = data.columns
    else:
        columns_to_color = range(data.shape[1])

    column_color_buttons = [
        {
            "args": [{
                "marker.color": [None, node_colors_color_variable],
                "marker.cmin": [None, color_variable_min],
                "marker.cmax": [None, color_variable_max],
                "hoverlabel.bgcolor": [None, node_colors_color_variable]
            }],
            "label": "color_variable",
            "method": "restyle"
        }
    ]

    for column in columns_to_color:
        if is_data_dataframe:
            column_values = data[column].to_numpy()
        else:
            column_values = data[:, column]
        node_summary_statistics = _get_node_summary(
            column_values, node_elements, summary_statistic
        )
        node_colors, min_node_summary, max_node_summary = \
            _get_node_colors(node_summary_statistics)
        node_colors = list(
            map(partial(_convert_to_hex, colorscale), node_colors)
        )

        column_color_buttons.append(
            {
                "args": [{
                    "marker.color": [None, node_colors],
                    "marker.cmin": [None, min_node_summary],
                    "marker.cmax": [None, max_node_summary],
                    "hoverlabel.bgcolor": [None, node_colors]
                }],
                "label": f"Column {column}",
                "method": "restyle"
            }
        )
    return column_color_buttons


def _infer_color_variable_kind(color_variable, data):
    """Determine whether color_variable is array, pandas dataframe, callable,
    or scikit-learn (fit-)transformer."""
    if hasattr(color_variable, "dtype") or hasattr(color_variable, "dtypes"):
        if len(color_variable) != len(data):
            raise ValueError(
                "color_variable and data must have the same length.")
        color_variable_kind = "scalars"
    elif hasattr(color_variable, "transform"):
        color_variable_kind = "transformer"
    elif hasattr(color_variable, "fit_transform"):
        color_variable_kind = "fit_transformer"
    elif callable(color_variable):
        color_variable_kind = "callable"
    elif color_variable is None:
        color_variable_kind = "none"
    else:  # Assume color_variable is a selection of columns
        color_variable_kind = "else"

    return color_variable_kind


def _get_node_summary_statistics(
        data, is_data_dataframe, node_elements, node_color_statistic,
        color_variable
):
    """Calculate values of node summary statistics."""
    color_variable_kind = _infer_color_variable_kind(color_variable, data)

    if color_variable_kind == "scalars":
        color_data = color_variable
    elif color_variable_kind == "transformer":
        color_data = color_variable.transform(data)
    elif color_variable_kind == "fit_transformer":
        color_data = color_variable.fit_transform(data)
    elif color_variable_kind == "callable":
        color_data = color_variable(data)
    elif color_variable_kind == "none":
        if is_data_dataframe:
            color_data = data.to_numpy()
        else:
            color_data = data
    else:
        if is_data_dataframe:
            color_data = data[color_variable].to_numpy()
        else:
            color_data = data[:, color_variable]

    return _get_node_summary(
        color_data, node_elements, summary_statistic=node_color_statistic)


def _get_node_colors(
        node_summary_statistics
):
    """Calculate node color values in the range [0, 1] from raw node summary
    statistics by performing a min-max scaling."""
    # Check if node_summary_statistics contains NaNs
    if any(np.logical_not(np.isfinite(node_summary_statistics))):
        from warnings import warn
        warn("NaN values detected among the node summary statistics! These "
             "values will be ignored in the color scale", RuntimeWarning)

    # Normalise in range [0, 1]
    nanmin = np.nanmin(node_summary_statistics)
    nanmax = np.nanmax(node_summary_statistics)
    return (node_summary_statistics - nanmin) / (nanmax - nanmin), nanmin, \
        nanmax


def _convert_to_hex(colormap, x):
    """Convert float `x` to hex values according to `colormap`"""
    return rgb2hex(get_cmap(colormap)(x))


def _calculate_graph_data(
        pipeline, data, is_data_dataframe, layout, layout_dim, color_variable,
        node_color_statistic, plotly_kwargs
):
    graph = pipeline.fit_transform(data)
    node_elements = graph["node_metadata"]["node_elements"]

    # Determine whether layout is an array of node positions
    is_layout_ndarray = hasattr(layout, "dtype")
    if is_layout_ndarray:
        if layout.shape[1] not in [2, 3]:
            raise ValueError(
                f"If an ndarray, `layout` must be 2D with 2 or 3 columns. "
                f"Array with {layout.shape[1]} columns passed."
            )
        node_pos = layout
    else:
        if layout_dim not in [2, 3]:
            raise ValueError(
                f"`layout_dim` must be either 2 or 3. {layout_dim} entered."
            )
        node_pos = graph.layout(layout, dim=layout_dim)

    # Determine whether node_colors is an array of node colors
    is_node_color_statistic_ndarray = hasattr(node_color_statistic, "dtype")
    if not (is_node_color_statistic_ndarray or callable(node_color_statistic)):
        raise ValueError("node_color_statistic must be a callable or ndarray.")

    # Compute the raw values of node summary statistics
    if is_node_color_statistic_ndarray:
        node_summary_statistics = node_color_statistic
    else:
        node_summary_statistics = _get_node_summary_statistics(
            data, is_data_dataframe, node_elements, node_color_statistic,
            color_variable
        )
    graph["node_metadata"]["node_summary_statistic"] = node_summary_statistics

    # Obtain node colors as the node summary statistics normalised in [0, 1]
    node_colors, color_variable_min, color_variable_max = _get_node_colors(
        node_summary_statistics)

    plot_options = {
        "node_trace": deepcopy(PLOT_OPTIONS_NODE_TRACE_DEFAULTS),
        "edge_trace": deepcopy(PLOT_OPTIONS_EDGE_TRACE_DEFAULTS)
    }

    plot_options["node_trace"]["text"] = _get_node_text(graph)
    plot_options["node_trace"]["marker"].update({
        "size": _get_node_size(node_elements),
        "sizeref": set_node_sizeref(node_elements),
        "cmin": color_variable_min,
        "cmax": color_variable_max
    })

    colorscale = plot_options["node_trace"]["marker"]["colorscale"]
    node_colors = list(map(partial(_convert_to_hex, colorscale), node_colors))
    plot_options["node_trace"]["marker"]["color"] = node_colors

    # if plotly_kwargs is not None:
    #     plot_options.update(plotly_kwargs)

    edge_x = list(
        reduce(
            operator.iconcat, map(
                lambda x: [node_pos[x[0]][0], node_pos[x[1]][0], None],
                graph.get_edgelist()
            ), []
        )
    )
    edge_y = list(
        reduce(
            operator.iconcat, map(
                lambda x: [node_pos[x[0]][1], node_pos[x[1]][1], None],
                graph.get_edgelist()
            ), []
        )
    )

    node_x = [node_pos[k][0] for k in range(graph.vcount())]
    node_y = [node_pos[k][1] for k in range(graph.vcount())]

    if layout_dim == 2:
        node_trace = go.Scatter(
            x=node_x, y=node_y, **plot_options["node_trace"]
        )

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, **plot_options["edge_trace"]
        )

    elif layout_dim == 3:
        node_z = [node_pos[k][2] for k in range(graph.vcount())]
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z, **plot_options["node_trace"]
        )

        edge_z = list(
            reduce(
                operator.iconcat, map(
                    lambda x: [node_pos[x[0]][2], node_pos[x[1]][2], None],
                    graph.get_edgelist()
                ), []
            )
        )

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, **plot_options["edge_trace"])

    return node_trace, edge_trace, node_elements, node_colors, \
        color_variable_min, color_variable_max
