"""Testing for plot_diagram."""
# License: GNU AGPLv3

import numpy as np

from gtda.plotting import plot_diagram


def test_plot_diagram_empty():
    """Test that plot_diagram does not crash on a diagram with no non-trivial
    points."""
    plot_diagram(np.array([[0., 0., 0.],
                           [0., 0., 1.]]))
