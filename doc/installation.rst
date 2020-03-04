############
Installation
############

.. _installation:

************
Dependencies
************

The latest stable version of giotto-tda requires:

- Python (>= 3.6)
- NumPy (>= 1.17.0)
- SciPy (>= 0.17.0)
- joblib (>= 0.13)
- scikit-learn (>= 0.22.0)
- python-igraph (>= 0.7.1.post6)
- matplotlib (>= 3.0.3)
- plotly (>= 4.4.1)
- ipywidgets (>= 7.5.1)

To run the examples, jupyter is required.


*****************
User installation
*****************

The simplest way to install giotto-tda is using ``pip``   ::

    pip install -U giotto-tda

If necessary, this will also automatically install all the above dependencies. Note: we recommend
upgrading ``pip`` to a recent version as the above may fail on very old versions.

Pre-release, experimental builds containing recently added features, and/or
bug fixes can be installed by running   ::

    pip install -U giotto-tda-nightly

The main difference between giotto-tda-nightly and the developer installation (see the section
on contributing, below) is that the former is shipped with pre-compiled wheels (similarly to the stable
release) and hence does not require any C++ dependencies. As the main library module is called ``gtda`` in
both the stable and nightly versions, giotto-tda and giotto-tda-nightly should not be installed in
the same environment.

**********************
Developer installation
**********************

For information about the developer installation, please refer to :ref:`Contributing guidelines <contrib>`.

