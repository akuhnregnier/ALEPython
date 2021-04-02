# -*- coding: utf-8 -*-
import numpy as np


class DummyModel:
    """A simple predictive model for testing purposes.

    Methods
    -------
    predict(X)
        Given input data `X`, predict response variable.

    fit(X, y)
        Dummy fit method.

    """

    @staticmethod
    def fit(X, y):
        pass

    @staticmethod
    def predict(X):
        return np.mean(X, axis=1)


class DummyLinearModel(DummyModel):
    """A simple linear effect with features 'a' and 'b'.

    Methods
    -------
    predict(X)
        Given input data `X`, predict response variable.

    fit(X, y)
        Dummy fit method.

    """

    @staticmethod
    def predict(X):
        return X["a"] + X["b"]


class DummyInteractionModel(DummyModel):
    """Interaction changes sign at b = 0.5.

    Assumes b is uniformly distributed in [0, 1).

    Methods
    -------
    predict(X)
        Given input data `X`, predict response variable.

    fit(X, y)
        Dummy fit method.

    """

    @staticmethod
    def predict(X):
        a = X["a"]
        b = X["b"]

        out = np.empty_like(a)

        mask = b < 0.5
        out[mask] = a[mask] * b[mask]
        mask = ~mask
        out[mask] = -a[mask] * (1 - b[mask])

        return out
