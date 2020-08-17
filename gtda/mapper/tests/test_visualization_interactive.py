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
        "initial": {"n_intervals": 10, "kind": "uniform", "overlap_frac": 0.1},
        "new": {"n_intervals": 15, "kind": "balanced", "overlap_frac": 0.2}
        },
    "clusterer": {
        "initial": {"affinity": "euclidean"},
        "new": {"affinity": "manhattan"}
        }
    }


def _get_widgets_by_trait(fig, key, val=None):
    widgets = []
    for k, v in fig.widgets.items():
        try:
            b = getattr(v, key) == val if val is not None else getattr(v, key)
            if b:
                widgets.append(fig.widgets[k])
        except (AttributeError, TypeError):
            continue
    return widgets


@pytest.mark.parametrize("clone_pipeline", [False, True])
@pytest.mark.parametrize("layout_dim", [2, 3])
def test_pipeline_cloned(clone_pipeline, layout_dim):
    """Verify that the pipeline is changed on interaction if and only if
    `clone_pipeline` is False (with `layout_dim` set to 2 or 3)."""
    # TODO: Monitor development of the ipytest project to convert these into
    # true notebook tests integrated with pytest
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
            new_param_value = values["new"][param_name]
            widgets = _get_widgets_by_trait(fig, "description", param_name)
            for w in widgets:
                w.set_state({'value': new_param_value})
            final_param_value_actual = \
                pipe.get_mapper_params()[f"{step}__{param_name}"]
            final_param_value_expected = \
                initial_param_value if clone_pipeline else new_param_value
            assert final_param_value_actual == final_param_value_expected
