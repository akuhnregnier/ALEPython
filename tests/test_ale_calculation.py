# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from alepython.ale import first_order_ale_quant, second_order_ale_quant

from .utils import interaction_predictor, linear_predictor


def test_linear():
    """We expect both X['a'] and X['b'] to have a linear relationship.

    There should be no second order interaction.

    """

    def linear(x, a, b):
        return a * x + b

    np.random.seed(1)

    N = int(1e5)
    X = pd.DataFrame({"a": np.random.random(N), "b": np.random.random(N)})

    # Test that the first order relationships are linear.
    for column in X.columns:
        quantiles, ale = first_order_ale_quant(linear_predictor, X, column, 10)
        p, V = np.polyfit(quantiles, ale, 1, cov=True)
        assert np.all(np.isclose(p, [1, -0.5], atol=1e-3))
        assert np.all(np.isclose(np.sqrt(np.diag(V)), 0))

    # Test that a second order relationship does not exist.
    ale_second_order = second_order_ale_quant(linear_predictor, X, X.columns, 21)[1]
    assert np.all(np.isclose(ale_second_order, 0))


def test_interaction():
    """Ensure that the method picks up a trivial interaction term."""
    np.random.seed(1)

    N = int(1e6)
    nbins = 3  # This needs to be an odd number to yield an even number of edges.

    b = np.linspace(0, 1, N)
    # Mirror a around b=0.5.
    a_comp = np.random.random(N // 2) * 2
    a = np.append(a_comp, a_comp[::-1])
    X = pd.DataFrame({"a": a, "b": b})

    quantiles_list, ale = second_order_ale_quant(
        interaction_predictor, X, X.columns, nbins
    )[:2]

    b_quantiles = quantiles_list[1]
    # Check that the quantiles are mirrored.
    assert np.allclose(
        b_quantiles[: (nbins + 1) // 2],
        1 - np.array(b_quantiles[(nbins + 1) // 2 :][::-1]),
    )
    # Check that the ALE is mirrored as expected from the nature of the
    # `interaction_predictor`. The error here (which we require to be below `atol`
    # (roughly)) is observed to decrease with `N`, as expected.
    assert np.allclose(
        ale[:, : (nbins + 1) // 2], ale[:, (nbins + 1) // 2 :][::-1, ::-1], atol=1e-3
    )
