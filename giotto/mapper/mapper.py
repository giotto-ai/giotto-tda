from sklearn.base import BaseEstimator, TransformerMixin


class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, lens=None, cover=None, clusterer=None):
        self.lens = lens
        self.cover = cover
        self.clusterer = clusterer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.lens_values = self.lens.fit_transform(X)
        return self
