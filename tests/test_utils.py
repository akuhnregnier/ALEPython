# -*- coding: utf-8 -*-
import re

import numpy as np
import pytest

from alepython.ale import _get_centres


def test_centres():
    x = np.array([1, 2, 3])
    c = _get_centres(x)
    assert c.dtype == np.float64
    assert not np.shares_memory(x, c)

    xf = x.astype("float16")
    cf = _get_centres(xf, inplace=True)
    assert np.allclose(xf[:-1], cf)
    assert cf.dtype == np.float16
    assert xf.dtype == np.float16
    assert np.shares_memory(xf, cf)

    np.random.seed(1)
    xa = np.random.random((10, 10, 10))
    with pytest.raises(
        ValueError,
        match=(
            "The axis entry '3' is out of bounds given the number of " "dimensions '3'"
        ),
    ):
        _get_centres(xa, axis=(2, 3))

    with pytest.raises(
        ValueError, match=re.escape("Duplicated entries were found in axis (1, 1, 2).")
    ):
        _get_centres(xa, axis=(1, -2, 2))

    xs = np.random.random((10, 10, 1))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected all dimensions specified by 'axis' (1, 2) to have a "
            "size of at least two, but got shape (10, 10, 1)."
        ),
    ):
        _get_centres(xs, axis=(1, 2))

    assert np.allclose(_get_centres(xs), (xs[:-1] + xs[1:]) / 2)
    assert np.allclose(_get_centres(xs, axis=1), (xs[:, :-1] + xs[:, 1:]) / 2)
    assert np.allclose(
        _get_centres(xa, axis=(0, 1)),
        (xa[1:, 1:] + xa[:-1, 1:] + xa[1:, :-1] + xa[:-1, :-1]) / 4,
    )
