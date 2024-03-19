"""Vector representations of persistence diagrams."""
# License: GNU AGPLv3

from typing import Callable
from numbers import Real

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from plotly.graph_objects import Figure, Scatter
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted

from ._metrics import betti_curves, landscapes, heats, \
    persistence_images, silhouettes
from ._utils import _subdiagrams, _bin, _make_homology_dimensions_mapping, \
    _homology_dimensions_to_sorted_ints
from ..base import PlotterMixin
from ..plotting import plot_heatmap
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params, check_diagrams


@adapt_fit_transform_docs
class BettiCurve(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Betti curves <betti_curve>` of persistence diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    considered separately, and their respective Betti curves are obtained by
    evenly sampling the :ref:`filtration parameter <filtered_complex>`.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    n_bins : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    samplings_ : dict
        For each number in `homology_dimensions_`, a discrete sampling of
        filtration parameters, calculated during :meth:`fit` according to the
        minimum birth and maximum death values observed across all samples.

    See also
    --------
    PersistenceLandscape, PersistenceEntropy, HeatKernel, Amplitude, \
    PairwiseDistance, Silhouette, PersistenceImage, \
    gtda.homology.VietorisRipsPersistence

    Notes
    -----
    The samplings in :attr:`samplings_` are in general different between
    different homology dimensions. This means that the j-th entry of a Betti
    curve in homology dimension q typically arises from a different parameter
    values to the j-th entry of a curve in dimension q'.

    """

    _hyperparameters = {
        "n_bins": {"type": int, "in": Interval(1, np.inf, closed="left")}
        }

    def __init__(self, n_bins=100, n_jobs=None):
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and, for each dimension separately,
        store evenly sample filtration parameter values in :attr:`samplings_`.
        Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)
        self._n_dimensions = len(self.homology_dimensions_)

        self._samplings, _ = _bin(
            X, "betti", n_bins=self.n_bins,
            homology_dimensions=self.homology_dimensions_
            )
        self.samplings_ = {dim: s.flatten()
                           for dim, s in self._samplings.items()}

        return self

    def transform(self, X, y=None):
        """Compute the Betti curves of diagrams in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_bins)
            Betti curves: one curve (represented as a one-dimensional array
            of integer values) per sample and per homology dimension seen
            in :meth:`fit`. Index i along axis 1 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagrams(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(betti_curves)(
                _subdiagrams(X[s], [dim], remove_dim=True),
                self._samplings[dim])
            for dim in self.homology_dimensions_
            for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt).\
            reshape(self._n_dimensions, len(X), -1).\
            transpose((1, 0, 2))

        return Xt

    def plot(self, Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of Betti curves arranged as in the
        output of :meth:`transform`. Include homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_bins)
            Collection of Betti curves, such as returned by :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in :attr:`homology_dimensions_`.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        check_is_fitted(self)

        homology_dimensions_mapping = _make_homology_dimensions_mapping(
            homology_dimensions, self.homology_dimensions_
            )

        layout_axes_common = {
            "type": "linear",
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            }
        layout = {
            "xaxis1": {
                "title": "Filtration parameter",
                "side": "bottom",
                "anchor": "y1",
                **layout_axes_common
                },
            "yaxis1": {
                "title": "Betti number",
                "side": "left",
                "anchor": "x1",
                **layout_axes_common
                },
            "plot_bgcolor": "white",
            "title": f"Betti curves from diagram {sample}"
            }

        fig = Figure(layout=layout)

        for ix, dim in homology_dimensions_mapping:
            fig.add_trace(Scatter(x=self.samplings_[dim],
                                  y=Xt[sample][ix],
                                  mode="lines",
                                  showlegend=True,
                                  name=f"H{dim}"))

        # Update traces and layout according to user input
        if plotly_params:
            fig.update_traces(plotly_params.get("traces", None))
            fig.update_layout(plotly_params.get("layout", None))

        return fig


@adapt_fit_transform_docs
class PersistenceLandscape(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence landscapes <persistence_landscape>` of persistence
    diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    considered separately, and layers of their respective persistence
    landscapes are obtained by evenly sampling the :ref:`filtration parameter
    <filtered_complex>`.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    n_layers : int, optional, default: ``1``
        How many layers to consider in the persistence landscape.

    n_bins : int, optional, default: ``100``
        The number of filtration parameter values, per available
        homology dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`.

    samplings_ : dict
        For each number in `homology_dimensions_`, a discrete sampling of
        filtration parameters, calculated during :meth:`fit` according to the
        minimum birth and maximum death values observed across all samples.

    See also
    --------
    BettiCurve, PersistenceEntropy, HeatKernel, Amplitude, PairwiseDistance, \
    Silhouette, PersistenceImage, gtda.homology.VietorisRipsPersistence

    Notes
    -----
    The samplings in :attr:`samplings_` are in general different between
    different homology dimensions. This means that the j-th entry of the
    k-layer of a persistence landscape in homology dimension q typically arises
    from a different parameter value to the j-th entry of a k-layer in
    dimension q'.

    """

    _hyperparameters = {
        "n_bins": {"type": int, "in": Interval(1, np.inf, closed="left")},
        "n_layers": {"type": int, "in": Interval(1, np.inf, closed="left")}
        }

    def __init__(self, n_layers=1, n_bins=100, n_jobs=None):
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and, for each dimension separately, store
        evenly sample filtration parameter values in :attr:`samplings_`. Then,
        return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)
        self._n_dimensions = len(self.homology_dimensions_)

        self._samplings, _ = _bin(
            X, "landscape", n_bins=self.n_bins,
            homology_dimensions=self.homology_dimensions_
            )
        self.samplings_ = {dim: s.flatten()
                           for dim, s in self._samplings.items()}

        return self

    def transform(self, X, y=None):
        """Compute the persistence landscapes of diagrams in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions * n_layers, \
            n_bins)
            Persistence landscapes, where ``n_homology_dimensions`` is the
            number of distinct homology dimensions seen in :meth:`fit`.
            Landscapes coming from different homology dimensions are stacked
            for each sample, so layer ``k`` of the landscape in the ``j``-th
            homology dimension in :attr:`homology_dimensions_` is
            ``X[i, n_homology_dimensions * j + k]``.

        """
        check_is_fitted(self)
        X = check_diagrams(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(landscapes)(_subdiagrams(X[s], [dim], remove_dim=True),
                                self._samplings[dim],
                                self.n_layers)
            for dim in self.homology_dimensions_
            for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs))
            )
        Xt = np.concatenate(Xt).\
            reshape(self._n_dimensions, len(X), self.n_layers, self.n_bins).\
            transpose((1, 0, 2, 3)).\
            reshape(len(X), self._n_dimensions * self.n_layers, self.n_bins)

        return Xt

    def plot(self, Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence landscapes arranged
        as in the output of :meth:`transform`. Include homology in multiple
        dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_layers, \
            n_bins
            Collection of persistence landscapes, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Homology dimensions for which the landscape should be plotted.
            ``None`` means plotting all dimensions present in
            :attr:`homology_dimensions_`.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        check_is_fitted(self)

        homology_dimensions_mapping = _make_homology_dimensions_mapping(
            homology_dimensions, self.homology_dimensions_
            )

        layout_axes_common = {
            "type": "linear",
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            }
        layout = {
            "xaxis1": {
                "side": "bottom",
                "anchor": "y1",
                **layout_axes_common
                },
            "yaxis1": {
                "side": "left",
                "anchor": "x1",
                **layout_axes_common
                },
            "plot_bgcolor": "white",
            }

        Xt_sample = Xt[sample]
        n_dims = len(self.homology_dimensions_)
        n_layers = Xt_sample.shape[0] // n_dims
        subplot_titles = [f"H{dim}" for _, dim in homology_dimensions_mapping]
        fig = make_subplots(rows=len(homology_dimensions_mapping), cols=1,
                            subplot_titles=subplot_titles)
        has_many_homology_dim = len(homology_dimensions_mapping) - 1
        for i, (inv_idx, dim) in enumerate(homology_dimensions_mapping):
            hom_dim_str = \
                f" ({subplot_titles[i]})" if has_many_homology_dim else ""
            for layer in range(n_layers):
                fig.add_trace(
                    Scatter(x=self.samplings_[dim],
                            y=Xt_sample[inv_idx * n_layers + layer],
                            mode="lines",
                            showlegend=True,
                            hoverinfo="none",
                            name=f"Layer {layer + 1}{hom_dim_str}"),
                    row=i + 1,
                    col=1
                    )

        fig.update_layout(
            title_text=f"Landscape representations of diagram {sample}",
            **layout.copy()
            )

        # Update traces and layout according to user input
        if plotly_params:
            fig.update_traces(plotly_params.get("traces", None))
            fig.update_layout(plotly_params.get("layout", None))

        return fig


@adapt_fit_transform_docs
class HeatKernel(BaseEstimator, TransformerMixin, PlotterMixin):
    """Convolution of persistence diagrams with a Gaussian kernel.

    Based on ideas in [1]_. Given a persistence diagram consisting of
    birth-death-dimension triples [b, d, q], subdiagrams corresponding to
    distinct homology dimensions are considered separately and regarded as sums
    of Dirac deltas. Then, the convolution with a Gaussian kernel is computed
    over a rectangular grid of locations evenly sampled from appropriate
    ranges of the :ref:`filtration parameter <filtered_complex>`. The
    same is done with the reflected images of the subdiagrams about the
    diagonal, and the difference between the results of the two convolutions is
    computed. The result can be thought of as a (multi-channel) raster image.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    sigma : float, optional default ``0.1``
        Standard deviation for Gaussian kernel.

    n_bins : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`.

    samplings_ : dict
        For each number in `homology_dimensions_`, a discrete sampling of
        filtration parameters, calculated during :meth:`fit` according to the
        minimum birth and maximum death values observed across all samples.

    See also
    --------
    BettiCurve, PersistenceLandscape, PersistenceEntropy, Amplitude, \
    PairwiseDistance, Silhouette, PersistenceImage, \
    gtda.homology.VietorisRipsPersistence

    Notes
    -----
    The samplings in :attr:`samplings_` are in general different between
    different homology dimensions. This means that the (i, j)-th pixel
    of an image in homology dimension q typically arises from a different
    pair of parameter values to the (i, j)-th pixel of an image in
    dimension q'.

    References
    ----------
    .. [1] J. Reininghaus, S. Huber, U. Bauer, and R. Kwitt, "A Stable
           Multi-Scale Kernel for Topological Machine Learning"; *2015 IEEE
           Conference on Computer Vision and Pattern Recognition (CVPR)*,
           pp. 4741--4748, 2015; `DOI: 10.1109/CVPR.2015.7299106
           <http://dx.doi.org/10.1109/CVPR.2015.7299106>`_.

    """

    _hyperparameters = {
        "n_bins": {"type": int, "in": Interval(1, np.inf, closed="left")},
        "sigma": {"type": Real, "in": Interval(0, np.inf, closed="neither")}
        }

    def __init__(self, sigma=0.1, n_bins=100, n_jobs=None):
        self.sigma = sigma
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and, for each dimension separately,
        store evenly sample filtration parameter values in :attr:`samplings_`.
        Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)
        self._n_dimensions = len(self.homology_dimensions_)

        self._samplings, self._step_size = _bin(
            X, "heat", n_bins=self.n_bins,
            homology_dimensions=self.homology_dimensions_
            )
        self.samplings_ = {dim: s.flatten()
                           for dim, s in self._samplings.items()}

        return self

    def transform(self, X, y=None):
        """Compute multi-channel raster images from diagrams in `X` by
        convolution with a Gaussian kernel and reflection about the diagonal.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_bins, \
            n_bins)
            Multi-channel raster images: one image per sample and one
            channel per homology dimension seen in :meth:`fit`. Index i
            along axis 1 corresponds to the i-th homology dimension in
            :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagrams(X, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs, mmap_mode="c")(delayed(
            heats)(_subdiagrams(X[s], [dim], remove_dim=True),
                   self._samplings[dim], self._step_size[dim], self.sigma)
            for dim in self.homology_dimensions_
            for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt).\
            reshape(self._n_dimensions, len(X), self.n_bins, self.n_bins).\
            transpose((1, 0, 2, 3))
        return Xt

    def plot(self, Xt, sample=0, homology_dimension_idx=0, colorscale="blues",
             plotly_params=None):
        """Plot a single channel –- corresponding to a given homology
        dimension -- in a sample from a collection of heat kernel images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_bins, \
            n_bins)
            Collection of multi-channel raster images, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be selected.

        homology_dimension_idx : int, optional, default: ``0``
            Index of the channel in the selected sample to be plotted. If `Xt`
            is the result of a call to :meth:`transform` and this index is i,
            the plot corresponds to the homology dimension given by the i-th
            entry in :attr:`homology_dimensions_`.

        colorscale : str, optional, default: ``"blues"``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        check_is_fitted(self)
        homology_dimension = self.homology_dimensions_[homology_dimension_idx]
        if homology_dimension != np.inf:
            homology_dimension = int(homology_dimension)
        x = self.samplings_[homology_dimension]

        return plot_heatmap(
            Xt[sample][homology_dimension_idx], x=x, y=x[::-1],
            colorscale=colorscale, origin="lower",
            title=f"Heat kernel representation of diagram {sample} in "
                  f"homology dimension {homology_dimension}",
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class PersistenceImage(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence images <TODO>` of persistence
    diagrams.

    Based on ideas in [1]_. Given a persistence diagram consisting of
    birth-death-dimension triples [b, d, q], the equivalent diagrams of
    birth-persistence-dimension [b, d-b, q] triples are computed and
    subdiagrams corresponding to distinct homology dimensions are considered
    separately and regarded as sums of Dirac deltas. Then, the convolution
    with a Gaussian kernel is computed over a rectangular grid of locations
    evenly sampled from appropriate ranges of the :ref:`filtration parameter
    <filtered_complex>`. The result can be thought of as a (multi-channel)
    raster image.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    sigma : float, optional default ``0.1``
        Standard deviation for Gaussian kernel.

    n_bins : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    weight_function : callable or None, default: ``None``
        Function mapping the 1D array of sampled persistence values (see
        :attr:`samplings_`) to a 1D array of weights. ``None`` is equivalent to
        passing ``numpy.ones_like``. More weight can be given to regions of
        high persistence by passing a monotonic function, e.g. the identity.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    effective_weight_function_ : callable
        Effective function corresponding to `weight_function`. Set in
        :meth:`fit`.

    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`.

    samplings_ : dict
        For each dimension in `homology_dimensions_`, a discrete sampling of
        birth parameters and one of persistence values, calculated during
        :meth:`fit` according to the minimum birth and maximum death values
        observed across all samples.

    weights_ : dict
        For each number in `homology_dimensions_`, an array of weights
        corresponding to the persistence values obtained from `samplings_`
        calculated during :meth:`fit` using the `weight_function`.

    See also
    --------
    BettiCurve, PersistenceLandscape, PersistenceEntropy, HeatKernel, \
    Amplitude, PairwiseDistance, gtda.homology.VietorisRipsPersistence

    Notes
    -----
    The samplings in :attr:`samplings_` are in general different between
    different homology dimensions. This means that the (i, j)-th pixel of a
    persistence image in homology dimension q typically arises from a different
    pair of parameter values to the (i, j)-th pixel of a persistence image in
    dimension q'.

    References
    ----------
    .. [1] H. Adams, T. Emerson, M. Kirby, R. Neville, C. Peterson, P. Shipman,
           S. Chepushtanova, E. Hanson, F. Motta, and L. Ziegelmeier,
           "Persistence Images: A Stable Vector Representation of Persistent
           Homology"; *Journal of Machine Learning Research 18, 1*,
           pp. 218-252, 2017; `DOI: 10.5555/3122009.3122017
           <http://dx.doi.org/10.5555/3122009.3122017>`_.

    """

    _hyperparameters = {
        "n_bins": {"type": int, "in": Interval(1, np.inf, closed="left")},
        "sigma": {"type": Real, "in": Interval(0, np.inf, closed="neither")},
        "weight_function": {"type": (Callable, type(None))}
        }

    def __init__(self, sigma=0.1, n_bins=100, weight_function=None,
                 n_jobs=None):
        self.sigma = sigma
        self.n_bins = n_bins
        self.weight_function = weight_function
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and, for each dimension separately,
        store evenly sample filtration parameter values in :attr:`samplings_`.
        Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])

        if self.weight_function is None:
            self.effective_weight_function_ = np.ones_like
        else:
            self.effective_weight_function_ = self.weight_function

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)
        self._n_dimensions = len(self.homology_dimensions_)

        self._samplings, self._step_size = _bin(
            X, "persistence_image",  n_bins=self.n_bins,
            homology_dimensions=self.homology_dimensions_
            )
        self.weights_ = {
            dim: self.effective_weight_function_(samplings_dim[:, 1])
            for dim, samplings_dim in self._samplings.items()
            }
        self.samplings_ = {dim: s.T for dim, s in self._samplings.items()}

        return self

    def transform(self, X, y=None):
        """Compute multi-channel raster images from diagrams in `X` by
        convolution with a Gaussian kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_bins, \
            n_bins)
            Multi-channel raster images: one image per sample and one channel
            per homology dimension seen in :meth:`fit`. Index i along axis 1
            corresponds to the i-th homology dimension in
            :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagrams(X, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs, mmap_mode="c")(
            delayed(persistence_images)(
                _subdiagrams(X[s], [dim], remove_dim=True),
                self._samplings[dim],
                self._step_size[dim],
                self.sigma,
                self.weights_[dim]
                )
            for dim in self.homology_dimensions_
            for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs))
            )
        Xt = np.concatenate(Xt).\
            reshape(self._n_dimensions, len(X), self.n_bins, self.n_bins).\
            transpose((1, 0, 2, 3))
        return Xt

    def plot(self, Xt, sample=0, homology_dimension_idx=0, colorscale="blues",
             plotly_params=None):
        """Plot a single channel -– corresponding to a given homology
        dimension -– in a sample from a collection of persistence images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_bins, \
            n_bins)
            Collection of multi-channel raster images, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be selected.

        homology_dimension_idx : int, optional, default: ``0``
            Index of the channel in the selected sample to be plotted. If `Xt`
            is the result of a call to :meth:`transform` and this index is i,
            the plot corresponds to the homology dimension given by the i-th
            entry in :attr:`homology_dimensions_`.

        colorscale : str, optional, default: ``"blues"``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        check_is_fitted(self)
        homology_dimension = self.homology_dimensions_[homology_dimension_idx]
        if homology_dimension != np.inf:
            homology_dimension = int(homology_dimension)
        samplings_x, samplings_y = self.samplings_[homology_dimension]

        return plot_heatmap(
            Xt[sample][homology_dimension_idx],
            x=samplings_x,
            y=samplings_y[::-1],
            colorscale=colorscale,
            origin="lower",
            title=f"Persistence image representation of diagram {sample} in "
                  f"homology dimension {homology_dimension}",
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class Silhouette(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Power-weighted silhouettes <weighted_silhouette>` of persistence
    diagrams.

    Based on ideas in [1]_. Given a persistence diagram consisting of
    birth-death-dimension triples [b, d, q], subdiagrams corresponding to
    distinct homology dimensions are considered separately, and their
    respective silhouettes are obtained by sampling the silhouette function
    over evenly spaced locations from appropriate ranges of the
    :ref:`filtration parameter <filtered_complex>`.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    power: float, optional, default: ``1.``
        The power to which persistence values are raised to define the
        :ref:`power-weighted silhouettes <weighted_silhouette>`.

    n_bins : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    samplings_ : dict
        For each number in `homology_dimensions_`, a discrete sampling of
        filtration parameters, calculated during :meth:`fit` according to the
        minimum birth and maximum death values observed across all samples.

    See also
    --------
    PersistenceLandscape, PersistenceEntropy, HeatKernel, Amplitude, \
    PairwiseDistance, BettiCurve, gtda.homology.VietorisRipsPersistence

    Notes
    -----
    The samplings in :attr:`samplings_` are in general different between
    different homology dimensions. This means that the j-th entry of a
    silhouette in homology dimension q typically arises from a different
    parameter values to the j-th entry of a curve in dimension q'.

    References
    ----------
    .. [1] F. Chazal, B. T. Fasy, F. Lecci, A. Rinaldo, and L. Wasserman,
           "Stochastic Convergence of Persistence Landscapes and Silhouettes";
           *In Proceedings of the thirtieth annual symposium on Computational
           Geometry*, Kyoto, Japan, 2014, pp. 474–483;
           `DOI: 10.1145/2582112.2582128
           <http://dx.doi.org/10.1145/2582112.2582128>`_.

    """

    _hyperparameters = {
        "power": {"type": Real, "in": Interval(0, np.inf, closed="right")},
        "n_bins": {"type": int, "in": Interval(1, np.inf, closed="left")}
        }

    def __init__(self, power=1., n_bins=100, n_jobs=None):
        self.power = power
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and, for each dimension separately,
        store evenly sample filtration parameter values in :attr:`samplings_`.
        Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)
        self._n_dimensions = len(self.homology_dimensions_)

        self._samplings, _ = _bin(
            X, "silhouette", n_bins=self.n_bins,
            homology_dimensions=self.homology_dimensions_
            )
        self.samplings_ = {dim: s.flatten()
                           for dim, s in self._samplings.items()}

        return self

    def transform(self, X, y=None):
        """Compute silhouettes of diagrams in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_bins)
            One silhouette (represented as a one-dimensional array)
            per sample and per homology dimension seen
            in :meth:`fit`. Index i along axis 1 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagrams(X)

        Xt = (Parallel(n_jobs=self.n_jobs)
              (delayed(silhouettes)(_subdiagrams(X[s], [dim], remove_dim=True),
                                    self._samplings[dim], power=self.power)
              for dim in self.homology_dimensions_
              for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs))))

        Xt = np.concatenate(Xt).\
            reshape(self._n_dimensions, len(X), -1).\
            transpose((1, 0, 2))
        return Xt

    def plot(self, Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of silhouettes arranged as in the
        output of :meth:`transform`. Include homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_bins)
            Collection of silhouettes, such as returned by :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in :attr:`homology_dimensions_`.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        check_is_fitted(self)

        homology_dimensions_mapping = _make_homology_dimensions_mapping(
            homology_dimensions, self.homology_dimensions_
            )

        layout_axes_common = {
            "type": "linear",
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            }
        layout = {
            "xaxis1": {
                "title": "Filtration parameter",
                "side": "bottom",
                "anchor": "y1",
                **layout_axes_common
                },
            "yaxis1": {
                "side": "left",
                "anchor": "x1",
                **layout_axes_common
                },
            "plot_bgcolor": "white",
            "title": f"Silhouette representation of diagram {sample}"
            }

        fig = Figure(layout=layout)

        for ix, dim in homology_dimensions_mapping:
            fig.add_trace(Scatter(x=self.samplings_[dim],
                                  y=Xt[sample][ix],
                                  mode="lines",
                                  showlegend=True,
                                  hoverinfo="none",
                                  name=f"H{dim}"))

        # Update traces and layout according to user input
        if plotly_params:
            fig.update_traces(plotly_params.get("traces", None))
            fig.update_layout(plotly_params.get("layout", None))

        return fig
