"""Graph layout functions and plotly layout functions."""
# License: GNU AGPLv3

import operator
from copy import deepcopy
from functools import reduce, partial

import numpy as np
import plotly.graph_objs as go

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
        node_ids, num_node_elements, node_summary_statistics
):
    return [
        f"Node ID: {node_id}<br>Node size: {num_elements}"
        f"<br>Summary statistic: {node_summary_statistic}"
        for node_id, num_elements, node_summary_statistic in zip(
            node_ids, num_node_elements, node_summary_statistics
            )
        ]


def _get_node_summary(data, node_elements, summary_statistic):
    return list(map(summary_statistic, (data[itr] for itr in node_elements)))


def _get_column_color_buttons(
        data, is_data_dataframe, node_elements, node_colors_color_variable,
        summary_statistic, hovertext_color_variable,
        colorscale_for_hoverlabel, n_sig_figs
):
    # TODO: Consider opting for just-in-time computation instead of computing
    # all node summary values ahead of time. Solution should preserve scroll
    # zoom functionality of 2D static visualisation.
    def replace_summary_statistic(current_hovertext, new_statistic):
        pos = current_hovertext.rfind(" ")
        new_hovertext = current_hovertext[:pos] + \
            f" {_round_to_n_sig_figs(new_statistic, n=n_sig_figs)}"
        return new_hovertext

    if is_data_dataframe:
        columns_to_color = data.columns
    else:
        columns_to_color = range(data.shape[1])

    column_color_buttons = [
        {
            "args": [{
                "marker.color": [None, node_colors_color_variable],
                "hovertext": [None, hovertext_color_variable]
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

        node_colors = _get_node_summary(
            column_values, node_elements, summary_statistic
        )
        hovertext = list(map(
            replace_summary_statistic, hovertext_color_variable,
            node_colors
        ))

        new_button = {
            "args": [{
                "marker.color": [None, node_colors],
                "hovertext": [None, hovertext]
            }],
            "label": f"Column {column}",
            "method": "restyle"
        }

        if colorscale_for_hoverlabel is not None:
            node_colors = np.asarray(node_colors)
            min_col = np.min(node_colors)
            max_col = np.max(node_colors)
            new_button["args"][0]["hoverlabel.bgcolor"] = [
                None,
                _get_colors_for_vals(node_colors, min_col, max_col,
                                     colorscale_for_hoverlabel)
            ]

        column_color_buttons.append(new_button)

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
        data, is_data_dataframe, node_elements, summary_statistic,
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

    return _get_node_summary(color_data, node_elements, summary_statistic)


def _calculate_graph_data(
        pipeline, data, is_data_dataframe, layout, layout_dim, color_variable,
        node_color_statistic, n_sig_figs, node_scale
):
    graph = pipeline.fit_transform(data)
    node_elements = graph["node_metadata"]["node_elements"]

    # Determine whether node_color_statistic is an array of node colors
    is_node_color_statistic_ndarray = hasattr(node_color_statistic, "dtype")
    if not (is_node_color_statistic_ndarray or callable(node_color_statistic)):
        raise ValueError(
            "`node_color_statistic` must be a callable or ndarray."
        )

    # Compute the raw values of node summary statistics
    if is_node_color_statistic_ndarray:
        node_colors_color_variable = node_color_statistic
    else:
        node_colors_color_variable = _get_node_summary_statistics(
            data, is_data_dataframe, node_elements, node_color_statistic,
            color_variable
        )

    # Load defaults for node and edge traces
    plot_options = {
        "node_trace": deepcopy(PLOT_OPTIONS_NODE_TRACE_DEFAULTS),
        "edge_trace": deepcopy(PLOT_OPTIONS_EDGE_TRACE_DEFAULTS)
    }

    # Update size and color of nodes
    node_sizes = _get_node_size(node_elements)
    plot_options["node_trace"]["marker"].update({
        "size": node_sizes,
        "sizeref": _set_node_sizeref(node_sizes, node_scale=node_scale),
        "color": node_colors_color_variable
    })

    # Generate hovertext
    node_ids = graph["node_metadata"]["node_id"]
    num_node_elements = map(len, graph["node_metadata"]["node_elements"])
    node_colors_round = map(
        partial(_round_to_n_sig_figs, n=n_sig_figs), node_colors_color_variable
    )
    plot_options["node_trace"]["hovertext"] = _get_node_text(
        node_ids, num_node_elements, node_colors_round
    )

    # Compute graph layout
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
        node_pos = np.asarray(graph.layout(layout, dim=layout_dim).coords)

    # Store x and y coordinates of edge endpoints
    edge_x = list(
        reduce(
            operator.iconcat, map(
                lambda e: [node_pos[e.source, 0], node_pos[e.target, 0],
                           None],
                graph.es
            ), []
        )
    )
    edge_y = list(
        reduce(
            operator.iconcat, map(
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
                operator.iconcat, map(
                    lambda e: [node_pos[e.source][2], node_pos[e.target][2],
                               None],
                    graph.es
                ), []
            )
        )
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, **plot_options["edge_trace"])

    return edge_trace, node_trace, node_elements, node_colors_color_variable


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
                            dtype=np.float_)
    elif colors[0][0] == "#":
        colors = np.asarray(list(map(_hex_to_rgb, colors)), dtype=np.float_)
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
