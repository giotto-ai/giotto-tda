from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from ._utils import ListFeatureUnion, func_from_callable_on_rows, identity
from .cover import CubicalCover
from .cluster import ParallelClustering
from .nerve import Nerve


global_pipeline_params = ('memory', 'verbose')
nodes_params = ('scaler', 'filter_func', 'cover')
clust_params = ('clusterer',)
nerve_params = ('min_intersection',)
nodes_params_prefix = 'pullback_cover__map_and_cover__'
clust_params_prefix = 'clustering__'
nerve_params_prefix = 'nerve__'


class MapperPipeline(Pipeline):
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
    return MapperPipeline(steps=all_steps, memory=memory, verbose=verbose)
