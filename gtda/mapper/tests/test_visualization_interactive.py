"""Testing for interactivity in Mapper plotting functions."""
# License: GNU AGPLv3

import numpy as np
import pytest
from gtda.mapper import CubicalCover, FirstSimpleGap, make_mapper_pipeline, \
    plot_interactive_mapper_graph

N = 50
d = 3
X = np.random.randn(N, d)

params = {
    "cover": {
        "initial": {"n_intervals": 10, "kind": "uniform",
                    "overlap_frac": 0.1},
        "final": {"n_intervals": 15, "kind": "balanced",
                  "overlap_frac": 0.2}
        },
    "clusterer": {"initial": {"affinity": "euclidean"},
                  "final": {"affinity": "manhattan"}}
    }


def _get_widget_by_trait(fig, key, val=None):
    for k, v in fig.widgets.items():
        try:
            b = getattr(v, key) == val if val is not None \
                else getattr(v, key)
            if b:
                return fig.widgets[k]
        except (AttributeError, TypeError):
            pass


def test_pipeline_cloned_2D():
    """Verify that the pipeline is not changed on interaction if
    `clone_pipeline` is True (with `layout_dim` set to 2)."""
    clone_pipeline = True
    layout_dim = 2

    pipe = make_mapper_pipeline(
        cover=CubicalCover(**params["cover"]["initial"]),
        clusterer=FirstSimpleGap(**params["clusterer"]["initial"])
        )
    fig = plot_interactive_mapper_graph(
        pipe, X, clone_pipeline=clone_pipeline, layout_dim=layout_dim
        )

    # Get relevant widgets and change their states, then check final values
    for step, values in params.items():
        for param_name, initial_param_value in values["initial"].items():
            w_text = _get_widget_by_trait(fig, 'description', param_name)
            final_param_value = values["final"][param_name]
            w_text.set_state({'value': final_param_value})
            final_pipe_param_value = \
                pipe.get_mapper_params()[f"{step}__{param_name}"]
            assert final_pipe_param_value == initial_param_value \
                if clone_pipeline else final_param_value


def test_pipeline_not_cloned_2D():
    """Verify that the pipeline is changed on interaction if `clone_pipeline`
    is False (with `layout_dim` set to 2)."""
    clone_pipeline = False
    layout_dim = 2

    pipe = make_mapper_pipeline(
        cover=CubicalCover(**params["cover"]["initial"]),
        clusterer=FirstSimpleGap(**params["clusterer"]["initial"])
        )
    fig = plot_interactive_mapper_graph(
        pipe, X, clone_pipeline=clone_pipeline, layout_dim=layout_dim
        )

    # Get relevant widgets and change their states, then check final values
    for step, values in params.items():
        for param_name, initial_param_value in values["initial"].items():
            w_text = _get_widget_by_trait(fig, 'description', param_name)
            final_param_value = values["final"][param_name]
            w_text.set_state({'value': final_param_value})
            final_pipe_param_value = \
                pipe.get_mapper_params()[f"{step}__{param_name}"]
            assert final_pipe_param_value == initial_param_value \
                if clone_pipeline else final_param_value


def test_pipeline_cloned_3D():
    """Verify that the pipeline is not changed on interaction if
    `clone_pipeline` is True (with `layout_dim` set to 3)."""
    clone_pipeline = True
    layout_dim = 3

    pipe = make_mapper_pipeline(
        cover=CubicalCover(**params["cover"]["initial"]),
        clusterer=FirstSimpleGap(**params["clusterer"]["initial"])
        )
    fig = plot_interactive_mapper_graph(
        pipe, X, clone_pipeline=clone_pipeline, layout_dim=layout_dim
        )

    # Get relevant widgets and change their states, then check final values
    for step, values in params.items():
        for param_name, initial_param_value in values["initial"].items():
            w_text = _get_widget_by_trait(fig, 'description', param_name)
            final_param_value = values["final"][param_name]
            w_text.set_state({'value': final_param_value})
            final_pipe_param_value = \
                pipe.get_mapper_params()[f"{step}__{param_name}"]
            assert final_pipe_param_value == initial_param_value \
                if clone_pipeline else final_param_value


def test_pipeline_not_cloned_3D():
    """Verify that the pipeline is changed on interaction if `clone_pipeline`
    is False (with `layout_dim` set to 3)."""
    clone_pipeline = False
    layout_dim = 3

    pipe = make_mapper_pipeline(
        cover=CubicalCover(**params["cover"]["initial"]),
        clusterer=FirstSimpleGap(**params["clusterer"]["initial"])
        )
    fig = plot_interactive_mapper_graph(
        pipe, X, clone_pipeline=clone_pipeline, layout_dim=layout_dim
        )

    # Get relevant widgets and change their states, then check final values
    for step, values in params.items():
        for param_name, initial_param_value in values["initial"].items():
            w_text = _get_widget_by_trait(fig, 'description', param_name)
            final_param_value = values["final"][param_name]
            w_text.set_state({'value': final_param_value})
            final_pipe_param_value = \
                pipe.get_mapper_params()[f"{step}__{param_name}"]
            assert final_pipe_param_value == initial_param_value \
                if clone_pipeline else final_param_value
