"""Testing for Mapper plotting functions."""
# License: GNU AGPLv3

from unittest import TestCase

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from sklearn.decomposition import PCA

from gtda.mapper import FirstSimpleGap, CubicalCover, make_mapper_pipeline, \
    plot_static_mapper_graph, plot_interactive_mapper_graph, \
    MapperInteractivePlotter


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

viridis_colorscale = ((0.0, "#440154"),
                      (0.1111111111111111, "#482878"),
                      (0.2222222222222222, "#3e4989"),
                      (0.3333333333333333, "#31688e"),
                      (0.4444444444444444, "#26828e"),
                      (0.5555555555555556, "#1f9e89"),
                      (0.6666666666666666, "#35b779"),
                      (0.7777777777777778, "#6ece58"),
                      (0.8888888888888888, "#b5de2b"),
                      (1.0, "#fde725"))

hsl_colorscale = ["hsl(19.0, 96.0%, 67.0%)",
                  "hsl(60.0, 100.0%, 87.0%)",
                  "hsl(203.0, 51.0%, 71.0%)"]


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


def test_color_data_invalid_length():
    pipe = make_mapper_pipeline()

    with pytest.raises(ValueError):
        plot_static_mapper_graph(pipe, X_arr, color_data=X_arr[:-1])


class DummyPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def transform(self, X):
        return PCA(n_components=self.n_components).fit_transform(X)


@pytest.mark.parametrize("color_features",
                         [PCA(n_components=2),
                          DummyPCA(n_components=2),
                          DummyPCA(n_components=2).transform])
def test_color_features_as_estimator_or_callable(color_features):
    pipe = make_mapper_pipeline()
    graph = pipe.fit_transform(X_arr)
    node_elements = graph.vs["node_elements"]

    pca = PCA(n_components=2)
    color_data_transformed = pca.fit_transform(X_arr)
    node_colors_color_features = \
        np.array([np.mean(color_data_transformed[itr, 0])
                  for itr in node_elements])

    fig = plot_static_mapper_graph(pipe, X_arr, color_data=X_arr,
                                   color_features=color_features)

    assert_almost_equal(fig.data[1].marker.color, node_colors_color_features)


def test_color_features_as_columns_fails_on_series():
    pipe = make_mapper_pipeline()

    with pytest.raises(ValueError, match="If `color_data` is a pandas series"):
        plot_static_mapper_graph(pipe, X_df, color_data=X_df["a"],
                                 color_features="a")


@pytest.mark.parametrize("color_features", [X_arr, X_df])
def test_invalid_color_features_types(color_features):
    pipe = make_mapper_pipeline()

    with pytest.raises(ValueError):
        plot_static_mapper_graph(pipe, X_arr,
                                 color_features=color_features)


@pytest.mark.parametrize(
    "color_data, color_features",
    [(X_df["a"] > 0.1, pd.get_dummies),
     (X_df["a"], lambda x: x**2),
     (X_df["a"], None),
     (X_arr, [0, 1])]
    )
def test_valid_color_data_transformed(color_data, color_features):
    """Test that no errors are thrown when pandas dataframes/series are passed
    as color_data and/or returned when applying color_features."""
    pipe = make_mapper_pipeline()
    plot_static_mapper_graph(pipe, X_arr, color_data=color_data,
                             color_features=color_features)


@pytest.mark.parametrize("is_2d", [False, True])
def test_node_color_statistic_as_ndarray(is_2d):
    pipe = make_mapper_pipeline()
    graph = pipe.fit_transform(X_arr)
    node_color_statistic_col_0 = np.arange(len(graph.vs))
    if is_2d:
        node_color_statistic = np.vstack([node_color_statistic_col_0,
                                          node_color_statistic_col_0]).T
    else:
        node_color_statistic = node_color_statistic_col_0

    fig = plot_static_mapper_graph(pipe, X_arr,
                                   node_color_statistic=node_color_statistic)

    assert np.array_equal(fig.data[1].marker.color, node_color_statistic_col_0)


def test_node_color_statistic_as_ndarray_wrong_length():
    pipe = make_mapper_pipeline()
    graph = pipe.fit_transform(X_arr)
    node_color_statistic = np.arange(len(graph.vs) + 1)

    with pytest.raises(ValueError):
        plot_static_mapper_graph(pipe, X_arr,
                                 node_color_statistic=node_color_statistic)


def test_invalid_type_node_color_statistic_static():
    pipe = make_mapper_pipeline()

    with pytest.raises(ValueError):
        plot_static_mapper_graph(pipe, X_arr, node_color_statistic="foo")


def test_invalid_node_color_statistic_interactive():
    pipe = make_mapper_pipeline()
    graph = pipe.fit_transform(X_arr)
    node_color_statistic = np.arange(len(graph.vs))
    with pytest.raises(ValueError):
        plot_interactive_mapper_graph(
            pipe, X_arr, node_color_statistic=node_color_statistic
            )


def test_invalid_color_features_as_array_of_indices():
    pipe = make_mapper_pipeline()
    with pytest.raises(ValueError):
        plot_static_mapper_graph(
            pipe, X_arr, color_data=X_arr,
            color_features=np.arange(X_arr.shape[1])
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
@pytest.mark.parametrize("color_data", [None, colors])
@pytest.mark.parametrize("node_color_statistic", [None, np.max])
def test_colors_same_2d_3d(X, color_data, node_color_statistic):
    pipe = make_mapper_pipeline()
    fig_2d = plot_static_mapper_graph(
        pipe, X, layout_dim=2, color_data=color_data,
        node_color_statistic=node_color_statistic
        )
    fig_3d = plot_static_mapper_graph(
        pipe, X, layout_dim=3, color_data=color_data,
        node_color_statistic=node_color_statistic
        )
    assert np.array_equal(fig_2d.data[1].marker.color,
                          fig_3d.data[1].marker.color)


@pytest.mark.parametrize("X, columns", [(X_arr, range(X_arr.shape[1])),
                                        (X_df, X_df.columns)])
@pytest.mark.parametrize("layout_dim", [2, 3])
def test_column_dropdown(X, columns, layout_dim):
    pipe = make_mapper_pipeline()
    fig = plot_static_mapper_graph(pipe, X, color_data=X,
                                   layout_dim=layout_dim)
    fig_buttons = fig.layout.updatemenus[0].buttons

    assert list(fig.data[1].marker.color) == \
           list(fig_buttons[0].args[0]["marker.color"][1])

    for i, col in enumerate(columns):
        fig_col = plot_static_mapper_graph(
            pipe, X, layout_dim=layout_dim, color_data=X, color_features=col
            )
        assert list(fig_col.data[1].marker.color) == \
               list(fig_buttons[i].args[0]["marker.color"][1])


def _get_size_from_hovertext(s):
    size_str = s.split("<br>")[3].split(": ")[1]
    return int(size_str)


class TestStaticPlot(TestCaseNoTemplate):

    def test_is_data_present(self):
        """Verify that what we see in the graph corresponds to
        the number of samples in the graph."""
        pipe = make_mapper_pipeline()
        fig = plot_static_mapper_graph(pipe, X_arr, color_data=colors,
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
        node_size_real = [len(node) for node in g.vs["node_elements"]]

        assert sum(node_sizes_vis) == sum(node_size_real)


def _get_widgets_by_trait(fig, key, val=None):
    """Returns a list of widgets containing attribute `key` which currently
    evaluates to the value `val`."""
    widgets = []
    for k, v in fig.widgets.items():
        try:
            b = getattr(v, key) == val if val is not None else getattr(v, key)
            if b:
                widgets.append(v)
        except (AttributeError, TypeError):
            continue

    return widgets


@pytest.mark.parametrize("X", [X_arr, X_df])
@pytest.mark.parametrize("color_data", [None, X_arr, X_df])
@pytest.mark.parametrize("layout_dim", [2, 3])
def test_interactive_plotter_attrs(X, color_data, layout_dim):
    """Simple tests on the attributes stored by MapperInteractivePlotter when
    plotting."""
    pipe = make_mapper_pipeline()
    plotter = MapperInteractivePlotter(pipe, X)
    plotter.plot(color_data=color_data, layout_dim=layout_dim)

    # 1 Test graph_
    graph = pipe.fit_transform(X)
    assert plotter.graph_.isomorphic(graph)

    # 2 Test pipeline_
    assert str(plotter.pipeline_) == str(pipe)

    # 3 Test color_features_
    if color_data is not None:
        color_data_transformed = color_data
    else:
        color_data_transformed = np.arange(len(X)).reshape(-1, 1)
    assert np.array_equal(plotter.color_features_, color_data_transformed)

    # 4 Test node_summaries_
    assert len(plotter.node_summaries_) == len(graph.vs)

    # 5 Test figure_
    static_fig = plot_static_mapper_graph(pipe, X, color_data=color_data,
                                          layout_dim=layout_dim)
    interactive_fig = plotter.figure_

    edge_trace_attrs = ["hoverinfo", "line", "name", "x", "y"]
    for attr in edge_trace_attrs:
        assert np.all(getattr(interactive_fig.data[0], attr) ==
                      getattr(static_fig.data[0], attr))

    # Excluding marker, which gets treated separately below
    node_trace_attrs = ["hoverinfo", "hovertext", "mode", "name", "x", "y"]
    for attr in node_trace_attrs:
        assert np.all(getattr(interactive_fig.data[1], attr) ==
                      getattr(static_fig.data[1], attr))

    marker_attrs = ["color", "colorbar", "colorscale", "line", "opacity",
                    "reversescale", "showscale", "size", "sizemin", "sizemode",
                    "sizeref"]
    for attr in marker_attrs:
        assert np.all(getattr(interactive_fig.data[1].marker, attr) ==
                      getattr(static_fig.data[1].marker, attr))


@pytest.mark.parametrize("clone_pipeline", [False, True])
def test_pipeline_cloned(clone_pipeline):
    """Verify that the pipeline is changed on interaction if and only if
    `clone_pipeline` is False."""
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
    fig = plot_interactive_mapper_graph(pipe, X_arr,
                                        clone_pipeline=clone_pipeline)

    # Get relevant widgets and change their states, then check final values
    for step, values in params.items():
        if step in ["cover", "clusterer"]:
            for param_name, initial_param_value in values["initial"].items():
                new_param_value = values["new"][param_name]
                widgets = _get_widgets_by_trait(fig, "description", param_name)
                for w in widgets:
                    w.set_state({"value": new_param_value})
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
                w.set_state({"value": new_param_value})
            final_param_value_actual = \
                pipe.get_mapper_params()[f"{step}"]
            final_param_value_expected = \
                initial_param_value if clone_pipeline else new_param_value
            assert final_param_value_actual == final_param_value_expected


def test_user_hoverlabel_bgcolor_interactive_3d():
    pipe = make_mapper_pipeline()
    plotter = MapperInteractivePlotter(pipe, X_arr)
    plotter.plot(layout_dim=3,
                 plotly_params={"node_trace": {"hoverlabel_bgcolor": "blue"}})

    assert plotter.figure_.data[1].hoverlabel.bgcolor == "blue"
