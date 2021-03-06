# -*- coding: utf-8 -*-
from contextlib import contextmanager
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from alepython import ale_plot

from .utils import DummyModel


@contextmanager
def assert_n_created_figures(n=1):
    """Assert that a given number of figures are created."""
    initial_fignums = plt.get_fignums()
    yield  # Do not catch exceptions (ie. no try except).
    new_fignums = set(plt.get_fignums()) - set(initial_fignums)
    n_new = len(new_fignums)
    assert n_new == n, "Expected '{n}' figure(s), got '{n_new}'.".format(
        n=n, n_new=n_new
    )


@pytest.mark.parametrize(
    "features,columns",
    (("a", ("a", "b")), (("a",), ("a", "b", "c")), (("a", "b"), ("a", "b", "c"))),
)
def test_model(features, columns):
    """Given a model with a predict method, a plot should be created."""
    np.random.seed(1)
    train_set = pd.DataFrame(np.random.random((100, len(columns))), columns=columns)
    with assert_n_created_figures():
        ale_plot(DummyModel(), train_set, features)
    # Clean up the created figure.
    plt.close()


@pytest.mark.parametrize(
    "features,columns", ((("a",), ("a", "b", "c")), (("a", "b"), ("a", "b", "c")))
)
def test_monte_carlo(features, columns):
    np.random.seed(1)
    train_set = pd.DataFrame(np.random.random((100, len(columns))), columns=columns)
    train_response = np.random.random((100,))
    with assert_n_created_figures():
        ale_plot(
            model=DummyModel(),
            train_set=train_set,
            features=features,
            train_response=train_response,
            monte_carlo=True,
        )
    # Clean up the created figure.
    plt.close()


def test_df_column_features():
    """Test the handling of the `features` argument.

    No matter the type of the `features` iterable, `ale_plot` should be able to select
    the right columns.

    """
    n_col = 3
    np.random.seed(1)
    train_set = pd.DataFrame(
        np.random.random((100, n_col)), columns=list(ascii_lowercase[:n_col])
    )
    with assert_n_created_figures():
        ale_plot(DummyModel(), train_set, train_set.columns[:1])
    # Clean up the created figure.
    plt.close()
