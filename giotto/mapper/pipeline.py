from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from .cluster import ParallelClustering
from .cover import CubicalCover
from .nerve import Nerve
from .utils._list_feature_union import ListFeatureUnion
from .utils.pipeline import func_from_callable_on_rows, identity

global_pipeline_params = ('memory', 'verbose')
nodes_params = ('scaler', 'filter_func', 'cover')
clust_params = ('clusterer',)
nerve_params = ('min_intersection',)
nodes_params_prefix = 'pullback_cover__map_and_cover__'
clust_params_prefix = 'clustering__'
nerve_params_prefix = 'nerve__'


class MapperPipeline(Pipeline):
    """Adapts :class:`sklearn.pipeline.Pipeline` to Mapper pipelines generated
    by ``giotto.mapper.make_mapper_pipeline``.

    To deal with the nested structure of the Pipeline objects returned by
    ``giotto.mapper.make_mapper_pipeline``, the convenience methods
    :meth:`get_mapper_params` and :meth:`set_mapper_params` allow for
    simple access to the parameters involved in the definition of a Mapper.

    Examples
    --------
    >>> from sklearn.cluster import DBSCAN
    >>> from sklearn.decomposition import PCA
    >>> from giotto.mapper import make_mapper_pipeline, CubicalCover
    >>> filter_func = PCA(n_components=2)
    >>> cover = CubicalCover()
    >>> clusterer = DBSCAN()
    >>> pipe = make_mapper_pipeline(filter_func=filter_func,
    ...                             cover=cover,
    ...                             clusterer=clusterer)
    >>> print(pipe.get_mapper_params()['clusterer__eps'])
    0.5
    >>> pipe.set_mapper_params(clusterer___eps=0.1)
    >>> print(pipe.get_mapper_params()['clusterer__eps'])
    0.1

    See also
    --------
    make_mapper_pipeline

    """
    # TODO abstract away common logic behind if statements in set_mapper_params
    def get_mapper_params(self, deep=True):
        pipeline_params = super().get_params(deep=True)
        return {**{param: pipeline_params[param]
                   for param in global_pipeline_params},
                **self._clean_dict_keys(pipeline_params, nodes_params_prefix),
                **self._clean_dict_keys(pipeline_params, clust_params_prefix),
                **self._clean_dict_keys(pipeline_params, nerve_params_prefix)}

    def set_mapper_params(self, **kwargs):
        mapper_nodes_kwargs = self._subset_kwargs(kwargs, nodes_params)
        mapper_clust_kwargs = self._subset_kwargs(kwargs, clust_params)
        mapper_nerve_kwargs = self._subset_kwargs(kwargs, nerve_params)
        if mapper_nodes_kwargs:
            super().set_params(
                **{nodes_params_prefix + key: mapper_nodes_kwargs[key]
                   for key in mapper_nodes_kwargs})
            [kwargs.pop(key) for key in mapper_nodes_kwargs]
        if mapper_clust_kwargs:
            super().set_params(
                **{clust_params_prefix + key: mapper_clust_kwargs[key]
                   for key in mapper_clust_kwargs})
            [kwargs.pop(key) for key in mapper_clust_kwargs]
        if mapper_nerve_kwargs:
            super().set_params(
                **{nerve_params_prefix + key: mapper_nerve_kwargs[key]
                   for key in mapper_nerve_kwargs})
            [kwargs.pop(key) for key in mapper_nerve_kwargs]
        super().set_params(**kwargs)
        return self

    @staticmethod
    def _subset_kwargs(kwargs, param_strings):
        return {key: value for key, value in kwargs.items()
                if key.startswith(param_strings)}

    @staticmethod
    def _clean_dict_keys(kwargs, prefix):
        return {
            key[len(prefix):]: kwargs[key]
            for key in kwargs
            if key.startswith(prefix) and not key.startswith(prefix + 'steps')}


def make_mapper_pipeline(scaler=MinMaxScaler(),
                         filter_func=PCA(n_components=2),
                         cover=CubicalCover(),
                         clusterer=DBSCAN(),
                         min_intersection=1,
                         n_jobs_outer=None,
                         **pipeline_kwargs):
    """Construct a MapperPipeline object according to the specified Mapper
    steps.

    All steps may be arbitrary scikit-learn Pipeline objects. The scaling
    and cover steps must be transformers implementing a ``fit_transform``
    method. The filter function step may be a transformer implementing a
    ``fit_transform``, or a callable acting on one-dimensional arrays -- in
    the latter case, a transformer is internally created whose
    ``fit_transform`` applies this callable independently on each row of the
    data. The clustering step need only implement a ``fit`` method storing
    clustering labels.

    Parameters
    ----------
    scaler : object, default: :class:`sklearn.preprocessing.MinMaxScaler`
        Scaling transformer.

    filter_func : object or callable, default: \
        :meth:`sklearn.decomposition.PCA`
        Filter function to apply to the scaled data.

    cover : object
        Covering transformer.

    clusterer : object
        Clustering object.

    min_intersection : int, optional, default: 1
        Minimum size of the intersection between clusters required for
        creating an edge in the final Mapper graph.

    n_jobs_outer : int or None, optional, default: ``None``
        The number of jobs to use in a joblib-parallel application of the
        clustering step to each pullback cover element. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Returns
    -------
    mapper_pipeline : :class:`MapperPipeline` object
        Output Mapper pipeline.

    Examples
    --------
    >>> from giotto.mapper import make_mapper_pipeline, Projection, \
    ... OneDimensionalCover, FirstHistogramGap
    >>> scaler = None
    >>> filter_func = Projection()
    >>> cover = OneDimensionalCover()
    >>> clusterer = FirstHistogramGap()
    >>> mapper = make_mapper_pipeline(scaler=None,
    ...                               filter_func=filter_func,
    ...                               cover=cover,
    ...                               clusterer=clusterer)
    >>> X = np.random.random((100, 2))
    >>> mapper_graph = mapper.fit_transform(X)

    See also
    --------
    MapperPipeline, giotto.mapper.method_to_transform, \
    giotto.mapper.cluster.ParallelClustering

    """
    memory = pipeline_kwargs.pop('memory', None)
    verbose = pipeline_kwargs.pop('verbose', False)
    if pipeline_kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(pipeline_kwargs.keys())[0]))

    # If filter_func is not a scikit-learn transformer, hope it as a
    # callable to be applied on each row separately. Then attempt to create a
    # FunctionTransformer object to implement this behaviour.
    if not hasattr(filter_func, 'transform'):
        ft_func = func_from_callable_on_rows(filter_func)
        _filter_func = FunctionTransformer(func=ft_func, validate=True)
    else:
        _filter_func = filter_func

    map_and_cover = Pipeline(
        steps=[('scaler', scaler if scaler is not None else identity()),
               ('filter_func', _filter_func), ('cover', cover)],
        verbose=verbose)
    all_steps = [
        ('pullback_cover', ListFeatureUnion(
            [('identity', identity()), ('map_and_cover', map_and_cover)])),
        ('clustering', ParallelClustering(
            clusterer=clusterer, n_jobs_outer=n_jobs_outer)),
        ('nerve', Nerve(min_intersection=min_intersection))]
    mapper_pipeline = MapperPipeline(
        steps=all_steps, memory=memory, verbose=verbose)
    return mapper_pipeline
