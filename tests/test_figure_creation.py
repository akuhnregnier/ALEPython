# -*- coding: utf-8 -*-
from contextlib import contextmanager
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from alepython import ale_plot


class SimpleModel:
    """A simple predictive model for testing purposes.

    Methods
    -------
    predict(X)
        Given input data `X`, predict response variable.

    """

    def predict(self, X):
        return np.mean(X, axis=1)


def simple_predictor(X):
    return np.mean(X, axis=1)


@contextmanager
def assert_n_created_figures(n=1):
    """Assert that a given number of figures are created."""
    initial_fignums = plt.get_fignums()
    yield  # Do not catch exceptions (ie. no try except).
    new_fignums = set(plt.get_fignums()) - set(initial_fignums)
    n_new = len(new_fignums)
    assert n_new == n, f"Expected '{n}' figure(s), got '{n_new}'."


def test_model():
    """Given a model with a predict method, a plot should be created."""
    plt.ion()  # Prevent plt.show() from blocking.
    n_cols = 3
    columns = list(ascii_lowercase[:n_cols])
    train_set = pd.DataFrame(np.random.random((100, 3)), columns=columns)
    with assert_n_created_figures():
        ale_plot(SimpleModel(), train_set, columns)
    # Clean up the created figure.
    plt.close()


def test_predictor():
    """Given a predictor function, a plot should be created."""
    plt.ion()  # Prevent plt.show() from blocking.
    n_cols = 3
    columns = list(ascii_lowercase[:n_cols])
    train_set = pd.DataFrame(np.random.random((100, 3)), columns=columns)
    with assert_n_created_figures():
        ale_plot(
            model=None,
            train_set=train_set,
            features=columns,
            predictor=simple_predictor,
        )
    # Clean up the created figure.
    plt.close()


def test_argument_handling():
    """Test that proper errors are raised."""
    with pytest.raises(ValueError, match=r".*'model'.*'predictor'.*"):
        ale_plot(model=None, train_set=pd.DataFrame([1]), features=[0])
