# coding: utf-8

from datatypes import Dataset

from sklearn.feature_selection import SelectKBest, f_classif

def univariate_feature_selection(n, ds):
    """
    Selects 'n' features in the dataset. Returns the Reduced Dataset
    n (int), ds (Dataset) -> Dataset
    """

    selector = SelectKBest(f_classif, n)
    selector.fit(ds.data, ds.target)
    features = selector.get_support(indices=True)
    return Dataset(selector.transform(ds.data), ds.target)

def lda():
    pass

def qda():
    pass
