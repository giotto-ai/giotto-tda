"""Testing for Mapper plotting functions."""
# License: GNU AGPLv3

from unittest import TestCase

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from gtda.mapper import FirstSimpleGap, CubicalCover, make_mapper_pipeline, \
    plot_static_mapper_graph, plot_interactive_mapper_graph


class TestCaseNoTemplate(TestCase):
    def setUp(self):
        pio.templates.default = None

    def tearDown(self):
        pio.templates.default = "plotly"


N = 50
d = 3
X_arr = np.random.randn(N, d)
X_df = pd.DataFrame(X_arr, columns=["a", "b", "c"])
colors = np.random.randint(0, 10, N)

viridis_colorscale = ((0.0, '#440154'),
                      (0.1111111111111111, '#482878'),
                      (0.2222222222222222, '#3e4989'),
                      (0.3333333333333333, '#31688e'),
                      (0.4444444444444444, '#26828e'),
                      (0.5555555555555556, '#1f9e89'),
                      (0.6666666666666666, '#35b779'),
                      (0.7777777777777778, '#6ece58'),
                      (0.8888888888888888, '#b5de2b'),
                      (1.0, '#fde725'))

hsl_colorscale = ['hsl(19.0, 96.0%, 67.0%)',
                  'hsl(60.0, 100.0%, 87.0%)',
                  'hsl(203.0, 51.0%, 71.0%)']


@pytest.mark.parametrize("X", [X_arr, X_df])
@pytest.mark.parametrize("layout_dim", [2, 3])
def test_valid_layout_dim(X, layout_dim):
    pipe = make_mapper_pipeline()
    fig = plot_static_mapper_graph(pipe, X, layout_dim=layout_dim)
    edge_trace = fig.data[0]
    assert hasattr(edge_trace, "x") and hasattr(edge_trace, "y")
    is_z_present = hasattr(edge_trace, "z")
    assert is_z_present if layout_dim == 3 else not is_z_present


@pytest.mark.parametrize("X", [X_arr, X_df])
@pytest.mark.parametrize("layout_dim", [1, 4])
def test_invalid_layout_dim(X, layout_dim):
    with pytest.raises(ValueError):
        pipe = make_mapper_pipeline()
        _ = plot_static_mapper_graph(pipe, X, layout_dim=layout_dim)


@pytest.mark.parametrize("X", [X_arr, X_df])
def test_invalid_layout_algorithm(X):
    with pytest.raises(KeyError):
        pipe = make_mapper_pipeline()
        _ = plot_static_mapper_graph(pipe, X, layout="foobar")


@pytest.mark.parametrize("X", [X_arr, X_df])
@pytest.mark.parametrize("layout_dim", [2, 3])
def test_valid_hoverlabel_bgcolor(X, layout_dim):
    pipe = make_mapper_pipeline()
    fig = plot_static_mapper_graph(
        pipe, X, layout_dim=layout_dim,
        plotly_params={"node_trace": {"hoverlabel_bgcolor": "white"}}
        )
    assert fig.data[1]["hoverlabel"]["bgcolor"] == "white"


@pytest.mark.parametrize("X", [X_arr, X_df])
def test_unsuitable_colorscale_for_hoverlabel_3d(X):
    pipe = make_mapper_pipeline()
    with pytest.warns(RuntimeWarning):
        _ = plot_static_mapper_graph(
            pipe, X, layout_dim=3,
            plotly_params={"node_trace": {"marker_colorscale": hsl_colorscale}}
            )


@pytest.mark.parametrize("X", [X_arr, X_df])
def test_valid_colorscale(X):
    pipe = make_mapper_pipeline()

    fig_2d = plot_static_mapper_graph(
        pipe, X, layout_dim=2,
        plotly_params={"node_trace": {"marker_colorscale": "blues"}}
        )
    fig_3d = plot_static_mapper_graph(
        pipe, X, layout_dim=3,
        plotly_params={"node_trace": {"marker_colorscale": "blues"}}
        )

    # Test that the custom colorscale is correctly applied both in 2d and in 3d
    marker_colorscale = fig_2d.data[1]["marker"]["colorscale"]
    marker_colorscale_3d = fig_3d.data[1]["marker"]["colorscale"]
    assert marker_colorscale == marker_colorscale_3d

    # Test that the default colorscale is "viridis" and that the custom one is
    # different
    fig_default = plot_static_mapper_graph(pipe, X)
    marker_colorscale_default = \
        fig_default.data[1]["marker"]["colorscale"]
    assert marker_colorscale_default == viridis_colorscale
    assert marker_colorscale != marker_colorscale_default


@pytest.mark.parametrize("X", [X_arr, X_df])
@pytest.mark.parametrize("color_variable", [None, colors])
@pytest.mark.parametrize("node_color_statistic", [None, np.max])
def test_colors_same_2d_3d(X, color_variable, node_color_statistic):
    pipe = make_mapper_pipeline()
    fig_2d = plot_static_mapper_graph(
        pipe, X, layout_dim=2, color_variable=color_variable,
        node_color_statistic=node_color_statistic
        )
    fig_3d = plot_static_mapper_graph(
        pipe, X, layout_dim=3, color_variable=color_variable,
        node_color_statistic=node_color_statistic
        )
    assert fig_2d.data[1].marker.color == fig_3d.data[1].marker.color


@pytest.mark.parametrize("X, columns", [(X_arr, range(X_arr.shape[1])),
                                        (X_df, X_df.columns)])
@pytest.mark.parametrize("layout_dim", [2, 3])
def test_color_by_column_dropdown(X, columns, layout_dim):
    pipe = make_mapper_pipeline()
    fig = plot_static_mapper_graph(
        pipe, X, layout_dim=layout_dim, color_by_columns_dropdown=True
        )
    fig_buttons = fig.layout.updatemenus[0].buttons

    assert list(fig.data[1].marker.color) == \
           list(fig_buttons[0].args[0]["marker.color"][1])

    for i, col in enumerate(columns):
        fig_col = plot_static_mapper_graph(
            pipe, X, layout_dim=layout_dim, color_variable=col
            )
        assert list(fig_col.data[1].marker.color) == \
               list(fig_buttons[i + 1].args[0]["marker.color"][1])


def _get_size_from_hovertext(s):
    size_str = s.split("<br>")[3].split(": ")[1]
    return int(size_str)


class TestStaticPlot(TestCaseNoTemplate):

    def test_is_data_present(self):
        """Verify that what we see in the graph corresponds to
        the number of samples in the graph."""
        pipe = make_mapper_pipeline()
        fig = plot_static_mapper_graph(pipe, X_arr,
                                       color_variable=colors,
                                       clone_pipeline=False)
        node_trace_x = fig.data[1].x
        node_trace_y = fig.data[1].y

        assert node_trace_x.shape[0] == node_trace_y.shape[0]

        num_nodes = node_trace_x.shape[0]
        assert len(X_arr) >= num_nodes

        fig_colors = fig.data[1].marker.color
        assert len(fig_colors) == num_nodes

    def test_cluster_sizes(self):
        """Verify that the total number of calculated clusters is equal to
        the number of displayed clusters."""
        pipe = make_mapper_pipeline(clusterer=FirstSimpleGap())
        fig = plot_static_mapper_graph(pipe, X_arr)
        node_trace = fig.data[1]

        node_sizes_vis = [_get_size_from_hovertext(ht) for ht in
                          node_trace.hovertext]

        g = pipe.fit_transform(X_arr)
        node_size_real = [len(node) for node in g.vs['node_elements']]

        assert sum(node_sizes_vis) == sum(node_size_real)


def _get_widgets_by_trait(fig, key, val=None):
    """Returns a list of widgets containing attribute `key` which currently
    evaluates to the value `val`."""
    widgets = []
    for k, v in fig.widgets.items():
        try:
            b = getattr(v, key) == val if val is not None else getattr(v, key)
            if b:
                widgets.append(fig.widgets[k])
        except (AttributeError, TypeError):
            continue
    return widgets


@pytest.mark.parametrize("X", [X_arr, X_df])
@pytest.mark.parametrize("clone_pipeline", [False, True])
@pytest.mark.parametrize("layout_dim", [2, 3])
@pytest.mark.parametrize("color_by_columns_dropdown", [True, False])
def test_pipeline_cloned(X, clone_pipeline, layout_dim,
                         color_by_columns_dropdown):
    """Verify that the pipeline is changed on interaction if and only if
    `clone_pipeline` is False (with `layout_dim` set to 2 or 3)."""
    # TODO: Monitor development of the ipytest project to convert these into
    # true notebook tests integrated with pytest
    params = {
        "cover": {
            "initial": {"n_intervals": 10, "kind": "uniform",
                        "overlap_frac": 0.1},
            "new": {"n_intervals": 15, "kind": "balanced", "overlap_frac": 0.2}
            },
        "clusterer": {
            "initial": {"affinity": "euclidean"},
            "new": {"affinity": "manhattan"}
            },
        "contract_nodes": {"initial": True, "new": False},
        "min_intersection": {"initial": 4, "new": 1},
        }

    pipe = make_mapper_pipeline(
        cover=CubicalCover(**params["cover"]["initial"]),
        clusterer=FirstSimpleGap(**params["clusterer"]["initial"]),
        contract_nodes=params["contract_nodes"]["initial"],
        min_intersection=params["min_intersection"]["initial"]
        )
    fig = plot_interactive_mapper_graph(
        pipe, X, clone_pipeline=clone_pipeline, layout_dim=layout_dim,
        color_by_columns_dropdown=color_by_columns_dropdown
        )

    # Get relevant widgets and change their states, then check final values
    for step, values in params.items():
        if step in ["cover", "clusterer"]:
            for param_name, initial_param_value in values["initial"].items():
                new_param_value = values["new"][param_name]
                widgets = _get_widgets_by_trait(fig, "description", param_name)
                for w in widgets:
                    w.set_state({'value': new_param_value})
                final_param_value_actual = \
                    pipe.get_mapper_params()[f"{step}__{param_name}"]
                final_param_value_expected = \
                    initial_param_value if clone_pipeline else new_param_value
                assert final_param_value_actual == final_param_value_expected
        else:
            initial_param_value = values["initial"]
            new_param_value = values["new"]
            widgets = _get_widgets_by_trait(fig, "description", step)
            for w in widgets:
                w.set_state({'value': new_param_value})
            final_param_value_actual = \
                pipe.get_mapper_params()[f"{step}"]
            final_param_value_expected = \
                initial_param_value if clone_pipeline else new_param_value
            assert final_param_value_actual == final_param_value_expected
