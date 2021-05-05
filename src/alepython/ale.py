# -*- coding: utf-8 -*-
"""ALE plotting for continuous or categorical features."""
import math
from collections.abc import Iterable
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from loguru import logger
from matplotlib import colors
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import cKDTree
from sklearn.base import clone
from tqdm.auto import tqdm

logger.disable("alepython")


__all__ = (
    "ale_plot",
    "first_order_ale_quant",
    "first_order_quant_plot",
    "second_order_ale_quant",
    "second_order_quant_plot",
)


def _sci_format(x, scilim=2):
    """Conditional string formatting.

    Parameters
    ----------
    x : array-like
        Values to format.
    scilim : float, optional
        If the decimal logarithm of `x` varies by more than `scilim`, scientific
        notation will be used.

    Returns
    -------
    formatted : array-like
        Formatted string values.

    """
    log_x = np.log10(np.abs(x))
    log_ptp = np.ptp(log_x)
    if log_ptp > scilim:
        return [
            np.format_float_scientific(v, precision=1, unique=False, exp_digits=1)
            for v in x
        ]
    else:
        min_log = np.min(log_x)
        if min_log < 0:
            dec = math.ceil(np.abs(min_log)) + (1 if log_ptp <= 1 else 0)
        else:
            dec = 0
        return [f"{v:0.{dec}f}" for v in x]


def _parse_features(features):
    """Standardise representation of column labels.

    Parameters
    ----------
    features : object
        One or more column labels.

    Returns
    -------
    features : array-like
        An array of input features.

    Examples
    --------
    >>> _parse_features(1)
    array([1])
    >>> _parse_features(('a', 'b'))
    array(['a', 'b'], dtype='<U1')

    """
    if isinstance(features, Iterable) and not isinstance(features, str):
        # If `features` is a non-string iterable.
        return np.asarray(features)
    else:
        # If `features` is not an iterable, or it is a string, then assume it
        # represents one column label.
        return np.asarray([features])


def _check_two_ints(values):
    """Retrieve two integers.

    Parameters
    ----------
    values : [2-iterable of] int
        Values to process.

    Returns
    -------
    values : 2-tuple of int
        The processed integers.

    Raises
    ------
    ValueError
        If more than 2 values are given.
    ValueError
        If the values are not integers.

    Examples
    --------
    >>> _check_two_ints(1)
    (1, 1)
    >>> _check_two_ints((1, 2))
    (1, 2)
    >>> _check_two_ints((1,))
    (1, 1)

    """
    if isinstance(values, (int, np.integer)):
        values = (values, values)
    elif len(values) == 1:
        values = (values[0], values[0])
    elif len(values) != 2:
        raise ValueError(
            "'{}' values were given. Expected at most 2.".format(len(values))
        )

    if not all(isinstance(n_bin, (int, np.integer)) for n_bin in values):
        raise ValueError(
            "All values must be an integer. Got types '{}' instead.".format(
                {type(n_bin) for n_bin in values}
            )
        )
    return values


def _full_slices(slices, axis=None, ndim=None):
    """Add `slice(None)` to the `slices` tuple to fully index the given dimensions.

    Parameters
    ----------
    slices : iterable of slice
        Slices to pad.
    axis : None or int or tuple of ints, optional
        Which axis the slice objects in `slices` refer to. By default, given slices
        are interpreted to refer to the first dimensions.
    ndim : int, optional
        How many dimensions should be indexed with the returned tuple. By default, the
        number of slices or one plus the largest axis value is used, whichever one is
        larger.

    Returns
    -------
    slices : tuple
        A tuple containing `slices`, padded with as many empty slices as needed to
        index `ndim` dimensions, inserted either before, after or in between the
        entries in `slices`, depending on `axis`.

    Raises
    ------
    ValueError
        If `axis` contains more elements than `slices`.
    ValueError
        If `ndim` is less than or equal to the largest `axis` value.

    Examples
    --------
    >>> _full_slices((slice(1),), axis=1, ndim=2)
    (slice(None, None, None), slice(None, 1, None))
    >>> _full_slices((slice(1), slice(2)), axis=(1, 2), ndim=3)
    (slice(None, None, None), slice(None, 1, None), slice(None, 2, None))

    """
    if axis is None:
        axis = tuple(range(len(slices)))
    elif isinstance(axis, (int, np.integer)):
        axis = (axis,)

    if len(axis) > len(slices):
        raise ValueError(
            f"Expected at most {len(slices)} entries in axis. Got axis {repr(axis)}.'"
        )

    if ndim is None:
        ndim = max((len(slices), max(axis) + 1))

    if ndim <= max(axis):
        raise ValueError(
            f"'ndim' {repr(ndim)} is incompatible with 'axis' {repr(axis)}."
        )

    out = [slice(None)] * ndim
    for s, ax in zip(slices, axis):
        out[ax] = s
    return tuple(out)


def _get_centres(x, axis=0, inplace=False):
    """Return bin centres from bin edges.

    Parameters
    ----------
    x : array-like
        The array to be averaged, where all dimensions referred to by `axis` must have
        a size of at least two.
    axis : None or int or tuple of ints, optional
        The axis or axes over which to compute the centres. By default the centres are
        computed over the first axis. Negative entries are converted to positive
        entries by adding the number of dimensions of `x`, so that `axis=-1` refers to
        the last dimension.
    inplace : bool, optional
        Compute the result in `x` without allocating a new array to hold the output. A
        copy may still have to be created, e.g. if `x` is not an `np.ndarray` (or
        subclass thereof) or does not have a floating point dtype. While not only
        potentially saving memory, this method may also be faster.

    Returns
    -------
    centres : array-like
        The centres of `x`, the shape of which is (I - 1, ...) for
        `x` with shape (I, ...) when `axis=0` (default).

        For multiple axes, e.g. `axis=(0, 1)`, the centres of `x` will have shape
        (I - 1, J - 1, ...) for `x` with shape (I, J, ...).

        If the input array already has a floating point dtype, `centres` will have the
        same dtype, and `numpy.float64` otherwise.

    Raises
    ------
    ValueError
        If an element of `axis` is out of bounds given the shape of `x`.
    ValueError
        If one of the dimensions specified by `axis` has a size of less than two.
    ValueError
        If `axis` contains duplicated elements.

    Examples
    --------
    >>> import numpy as np
    >>> _get_centres(np.array([0, 1, 2, 3]))
    array([0.5, 1.5, 2.5])
    >>> _get_centres([[1,2,3], [1,5,3]], axis=(0, 1))
    array([[2.25, 3.25]])
    >>> _get_centres([[1,2,3], [1,5,3]], axis=(1, 0))
    array([[2.25, 3.25]])
    >>> x = np.array([0, 1, 2, 3])
    >>> c = _get_centres(x)
    >>> np.all(x[:-1] == c)
    False
    >>> # Since `x` does not have a floating point dtype, this will not succeed.
    >>> # _get_centres(x, inplace=True)
    >>> xf = x.astype('float16')
    >>> c = _get_centres(xf, inplace=True)
    >>> c
    array([0.5, 1.5, 2.5], dtype=float16)
    >>> np.all(xf[:-1] == c)
    True

    """
    if axis is None:
        axis = (0,)
    elif isinstance(axis, (int, np.integer)):
        axis = (axis,)

    dtype = np.float64
    if isinstance(x, np.ndarray):
        if x.dtype.kind == "f":
            # Do not change the dtype if it is already a floating point type.
            dtype = None

    if inplace:
        x = np.asanyarray(x, dtype=dtype)
    else:
        # Copy the array.
        x = np.array(x, subok=True, dtype=dtype)

    ndim = len(x.shape)

    # Convert negative axis entries.
    axis = tuple(ax if ax >= 0 else ndim + ax for ax in axis)

    # Check axis bounds.
    if max(axis) >= ndim:
        raise ValueError(
            f"The axis entry '{max(axis)}' is out of bounds given the number of "
            f"dimensions '{ndim}'"
        )

    if len(set(axis)) != len(axis):
        raise ValueError(f"Duplicated entries were found in axis {repr(axis)}.")

    if any(x.shape[ax] < 2 for ax in axis):
        raise ValueError(
            f"Expected all dimensions specified by 'axis' {repr(axis)} to have a "
            f"size of at least two, but got shape {repr(x.shape)}."
        )

    nd_slices = partial(_full_slices, ndim=ndim)

    for ax in axis:
        x[nd_slices((slice(-1),), axis=ax)] += x[nd_slices((slice(1, None),), axis=ax)]
        # Remove the now extraneous elements from the current dimension, since it has
        # already been aggregated.
        x = x[nd_slices((slice(-1),), axis=ax)]

    x /= 2 * len(axis)
    return x


def _ax_title(ax, title, subtitle=""):
    """Add title to axis.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes object to add title to.
    title : str
        Axis title.
    subtitle : str, optional
        Sub-title for figure. Will appear one line below `title`.

    """
    ax.set_title("\n".join((title, subtitle)))


def _ax_labels(ax, xlabel=None, ylabel=None):
    """Add labels to axis.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes object to add labels to.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.

    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _ax_quantiles(ax, quantiles, twin="x"):
    """Plot quantiles of a feature onto axis.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis to modify.
    quantiles : array-like
        Quantiles to plot.
    twin : {'x', 'y'}, optional
        Select the axis for which to plot quantiles.

    Returns
    -------
    ax_mod : matplotlib Axes
        Axis containing the quantile ticks.

    Raises
    ------
    ValueError
        If `twin` is not one of 'x' or 'y'.

    """
    if twin not in ("x", "y"):
        raise ValueError("'twin' should be one of 'x' or 'y'.")

    logger.debug("Quantiles: {}.", quantiles)

    # Duplicate the 'opposite' axis so we can define a distinct set of ticks for the
    # desired axis (`twin`).
    ax_mod = ax.twiny() if twin == "x" else ax.twinx()

    # Set the new axis' ticks for the desired axis.
    getattr(ax_mod, "set_{twin}ticks".format(twin=twin))(quantiles)
    # Set the corresponding tick labels.

    # Calculate tick label percentage values for each quantile (bin edge).
    percentages = (
        100 * np.arange(len(quantiles), dtype=np.float64) / (len(quantiles) - 1)
    )

    # If there is a fractional part, add a decimal place to show (part of) it.
    fractional = (~np.isclose(percentages % 1, 0)).astype("int8")

    getattr(ax_mod, f"set_{twin}ticklabels")(
        [
            f"{percent:0.{format_fraction}f}%"
            for percent, format_fraction in zip(percentages, fractional)
        ],
        color="#545454",
        fontsize=7,
    )
    getattr(ax_mod, f"set_{twin}lim")(getattr(ax, f"get_{twin}lim")())
    return ax_mod


def first_order_quant_plot(quantiles, ale, ax=None, **kwargs):
    """First order ALE plot.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis to plot onto. If None (default), the current axis will be used (a new one
        will be created if needed).
    quantiles : array-like
        ALE quantiles.
    ale : array-like
        ALE to plot.
    **kwargs : plot properties, optional
        Additional keyword parameters are passed to `ax.plot`.

    Returns
    -------
    axes : dict of matplotlib Axes
        Axes containing the ALE plot and potentially other elements depending on the
        parameters given.

    """
    if "marker" not in kwargs:
        kwargs["marker"] = "o"
    if "ms" not in kwargs:
        kwargs["ms"] = 3

    if ax is None:
        ax = plt.gca()

    ax.plot(quantiles, ale, **kwargs)
    return {"ale": ax}


class MidpointNormalize(colors.Normalize):
    def __init__(self, *args, midpoint=None, **kwargs):
        self.midpoint = midpoint
        super().__init__(*args, **kwargs)

    def __call__(self, value, clip=None):
        # Simple mapping between the color range halves and vmin and vmax.
        result, _ = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), mask=result.mask)


def second_order_quant_plot(
    quantiles_list, ale, samples=None, fig=None, ax=None, **kwargs
):
    """Second order ALE plot.

    Parameters
    ----------
    quantiles_list : (2, M + 1, N + 1) array-like
        ALE quantiles for the first (`quantiles_list[0]`) and second
        (`quantiles_list[1]`) features.
    ale : (M + 1, N + 1) masked array
        ALE to plot (see `second_order_ale_quant()`).
    samples : (M, N) array, optional
        The number of samples in each quantile bin. Used to determine empty cells.
    fig : matplotlib Figure
        Figure to plot onto. Required to plot colorbar. The current figure and axes
        will be used if both `fig` and `ax` are None (default). Will be used to create
        `ax` if only `ax` is `None`.
    ax : matplotlib Axes
        Axis to plot onto. If None (default), the current axis will be used (a new one
        will be created if needed).

    Returns
    -------
    fig : matplotlib Figure
        ALE plot figure.
    axes : dict of matplotlib Axes
        Axes containing the ALE plot and other elements like a colorbar depending on
        the parameters given.

    Other Parameters
    ----------------
    kind : {'contourf', 'grid', 'gridcontour'}, optional
        Type of visualisation. 'contourf' first interpolates `ale` before plotting the
        contours on `levels` levels. By default ('gridcontour'), raw values are
        plotted on a grid and overlaid with a labelled contours with `levels` levels.
    masking : {'clip', bool}, optional
        If True (default), the `ale.mask` will be used to ignore masked points. If
        False, the mask will be discarded. If 'clip', 'vmin' and 'vmax' are set using
        unmasked data only.
    min_samples : int, optional
        Cells where `samples` <= `min_samples` will be indicated using grey
        rectangles (see `indicate_empty`). By default, `min_samples` is 1, meaning
        that only quantile bins with no data are considered invalid.
    indicate_empty : {float, bool}, optional
        By default (`indicate_empty=0.4`), empty quantile bins are shaded grey if
        `samples` is given. Higher values (up to 1) will increase the shading
        opacity. Giving False will disable the indication of empty bins.
    n_interp : [2-iterable of] int, optional
        Applies to `kind` 'contourf' and 'gridcontour'. The number of interpolated
        samples generated from `ale` prior to contour plotting. Two integers may be
        given to specify different interpolation steps for the two features. If 0 is
        given, interpolation is disabled. By default three times the number of bins
        along each axis are used.
    interp_mask_thres : float, optional
        If `masking` is True, `ale.mask` is interpolated for `kind=contourf`. The
        bilinear interpolation yields fractional values ranging from 0 (all source
        points valid) to 1 (all source points invalid) based on the original point
        validity and distance to the target point. If the interpolated mask value
        exceeds `interp_mask_thres` (0.5 by default), the resulting interpolated point
        will be masked.
    levels : int, optional
        Applies to `kind` 'contourf' and 'gridcontour'. The number of contour levels.
        The default is 10.
    colorbar : {False, 'centered', 'symmetric'}, optional
        If `False`, do not plot a colorbar. If 'centered' or 'symmetric', the colorbar
        is centered at 0. If 'centered', colors are chosen according to the range
        [0, `min(ale)`] for negative values and [0, `max(ale)`] for positive values.
        If 'symmetric',the equivalent ranges are [0, `-max(abs(ale))`] and
        [0, `max(abs(ale))`]. The latter is useful when wanting to distinguish not
        only the sign, but also relative magnitudes of `ale`. Both options change
        `cmap` to 'bwr'. Relevant parameters in `kwargs` have precedence.
    colorbar_kwargs : dict, optional
        Parameters to pass to `fig.colorbar()`.
    **kwargs : optional
        Depending on `kind`, additional keyword parameters are passed to
        `ax.contourf()` ('contourf') or `ax.pcolormesh()` ('grid', 'gridcontour').

    Raises
    ------
    ValueError
        If `fig` is `None`, but `ax` is not None.
    ValueError
        If `n_interp` values are not integers.
    ValueError
        If more than 2 values are given for `n_interp`.
    ValueError
        If an unknown `kind` is given.

    """
    # Record the originally submitted arguments.
    orig_kw = locals()["kwargs"].copy()

    kind = kwargs.pop("kind", "gridcontour")
    masking = kwargs.pop("masking", True)
    min_samples = kwargs.pop("min_samples", 1)
    indicate_empty = kwargs.pop("indicate_empty", 0.4)
    levels = kwargs.pop("levels", 10)
    colorbar = kwargs.pop("colorbar", True)
    colorbar_kwargs = kwargs.pop("colorbar_kwargs", {})

    if "zorder" not in kwargs:
        kwargs["zorder"] = 1

    if fig is None and ax is None:
        fig, ax = plt.gcf(), plt.gca()
    elif fig is not None and ax is None:
        ax = fig.add_subplot(111)
    elif fig is None and ax is not None:
        raise ValueError("An axis ('ax') was supplied without a figure ('fig').")

    axes = {"ale": ax}

    if not masking:
        # Ignore the ALE mask.
        ale.mask = False
    elif masking == "clip":
        # Set vmin, vmax from masked data, then discard the mask.
        if "vmin" not in kwargs:
            kwargs["vmin"] = np.min(ale)
        if "vmax" not in kwargs:
            kwargs["vmax"] = np.max(ale)
        ale.mask = False

    if samples is not None:
        missing = samples < min_samples
    else:
        missing = False

    if np.any(missing):
        if not np.isclose(indicate_empty, 0):
            # Add rectangles to indicate cells without samples.
            for i, j in zip(*np.where(missing)):
                ax.add_patch(
                    Rectangle(
                        [quantiles_list[0][i], quantiles_list[1][j]],
                        quantiles_list[0][i + 1] - quantiles_list[0][i],
                        quantiles_list[1][j + 1] - quantiles_list[1][j],
                        linewidth=1,
                        edgecolor="none",
                        facecolor="k",
                        alpha=indicate_empty,
                        zorder=3,
                    )
                )

    if "norm" not in kwargs:
        if colorbar in ("centered", "symmetric"):
            kwargs["norm"] = MidpointNormalize(midpoint=0)

            if colorbar == "symmetric" and not {"vmin", "vmax"}.intersection(orig_kw):
                # Only modify 'vmin' and 'vmax' if they were not given originally.
                kwargs["vmax"] = max(
                    (
                        abs(kwargs.get("vmin", np.min(ale))),
                        kwargs.get("vmax", np.max(ale)),
                    )
                )
                kwargs["vmin"] = -kwargs["vmax"]

            # Change the default colormap to pronounce deviations from 0.
            if "cmap" not in kwargs:
                kwargs["cmap"] = "bwr"

    if kind == "contourf":
        n_interp = kwargs.pop("n_interp", None)
        interp_mask_thres = kwargs.get("interp_mask_thres", 0.5)

        if n_interp is not None:
            ns = list(_check_two_ints(n_interp))
            for i, (n, centres) in enumerate(zip(ns, quantiles_list)):
                if n == 0:
                    # Disable interpolation.
                    ns[i] = len(centres)
            n_x, n_y = ns
        else:
            n_x, n_y = 3 * len(quantiles_list[0]), 3 * len(quantiles_list[1])
        x = np.linspace(quantiles_list[0][0], quantiles_list[0][-1], n_x)
        y = np.linspace(quantiles_list[1][0], quantiles_list[1][-1], n_y)

        X, Y = np.meshgrid(x, y, indexing="xy")
        get_ale_interp = scipy.interpolate.interp2d(*quantiles_list, ale.data.T)

        # Bilinear interpolation is applied to the mask (as floats). The resulting
        # values are compared to the `interp_mask_thres` threshold to calculate
        # the interpolated mask.
        get_mask_interp = scipy.interpolate.interp2d(
            *quantiles_list, ale.mask.T.astype("float64")
        )

        ale_interp = get_ale_interp(x, y)
        mask_interp = get_mask_interp(x, y)
        masked_ale_interp = np.ma.MaskedArray(
            ale_interp, mask=mask_interp > interp_mask_thres
        )
        img = ax.contourf(X, Y, masked_ale_interp, levels=levels, **kwargs)
    elif kind in ("grid", "gridcontour"):
        # Generate cell edges from the quantiles.
        cell_edges_list = []
        for quantiles in quantiles_list:
            cell_edges_list.append(
                np.asarray([quantiles[0], *_get_centres(quantiles), quantiles[-1]])
            )
        # Plot the grid.
        img = ax.pcolormesh(*cell_edges_list, ale.T, **kwargs)

        if kind == "gridcontour":
            # Plot contour lines over the grid.
            cs = ax.contour(
                *quantiles_list, ale.T, levels=levels, colors="k", alpha=0.8, zorder=2
            )
            ax.clabel(cs)
    else:
        raise ValueError(f"Unknown 'kind' {repr(kind)}.")

    if colorbar:
        # Get the colorbar axis.
        axes["colorbar"] = fig.colorbar(img, **colorbar_kwargs).ax

    return fig, axes


def _get_quantiles(train_set, feature, bins):
    """Get quantiles from a feature in a dataset.

    Parameters
    ----------
    train_set : pandas.core.frame.DataFrame
        Dataset containing feature `feature`.
    feature : column label
        Feature for which to calculate quantiles.
    bins : int
        The number of quantiles is calculated as `bins + 1`.

    Returns
    -------
    quantiles : array-like
        Quantiles.
    bins : int
        Number of bins, `len(quantiles) - 1`. This may be lower than the original
        `bins` if identical quantiles were present.

    Raises
    ------
    ValueError
        If `bins` is not an integer.

    Notes
    -----
    When using this definition of quantiles in combination with a half open interval
    (lower quantile, upper quantile], care has to taken that the smallest observation
    is included in the first bin. This is handled transparently by `np.digitize`.

    Floating point errors can cause misassignment of quantiles in some cases.

    """
    if not isinstance(bins, (int, np.integer)):
        raise ValueError(
            "Expected integer 'bins', but got type '{}'.".format(type(bins))
        )
    quantiles = np.unique(
        np.quantile(
            train_set[feature], np.linspace(0, 1, bins + 1), interpolation="lower"
        )
    )
    bins = len(quantiles) - 1
    return quantiles, bins


def first_order_ale_quant(predictor, train_set, feature, bins, include_mean=False):
    """Estimate the first-order ALE function for single continuous feature data.

    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which the model was trained.
    feature : column label
        Feature name. A single column label.
    bins : int
        This defines the number of bins to compute. The effective number of bins may
        be less than this as only unique quantile values of train_set[feature] are
        used.
    include_mean : bool, optional
        Add the mean of all predictions to the output array.

    Returns
    -------
    quantiles : array-like
        Quantiles used.
    ale : array-like
        The first order ALE.

    """
    quantiles, _ = _get_quantiles(train_set, feature, bins)
    logger.debug("Quantiles: {}.", quantiles)

    # Define the bins the feature samples fall into. Shift and clip to ensure we are
    # getting the index of the left bin edge and the smallest sample retains its index
    # of 0.
    indices = np.clip(
        np.digitize(train_set[feature], quantiles, right=True) - 1, 0, None
    )

    # Assign the feature quantile values to two copied training datasets, one for each
    # bin edge. Then compute the difference between the corresponding predictions
    predictions = []
    for offset in range(2):
        mod_train_set = train_set.copy()
        mod_train_set[feature] = quantiles[indices + offset]
        predictions.append(predictor(mod_train_set))
    # The individual effects.
    effects = predictions[1] - predictions[0]

    # Average these differences within each bin.
    index_groupby = pd.DataFrame({"index": indices, "effects": effects}).groupby(
        "index"
    )

    mean_effects = index_groupby.mean().to_numpy().flatten()

    ale = np.array([0, *np.cumsum(mean_effects)])

    # Centre the ALE by subtracting its expectation value.
    ale -= np.sum(_get_centres(ale) * index_groupby.size() / train_set.shape[0])

    if include_mean:
        ale += sum(np.mean(preds) for preds in predictions) / 2

    return quantiles, ale


def second_order_ale_quant(
    predictor,
    train_set,
    features,
    bins,
    include_mean=False,
    n_jobs=1,
    n_neighbour=10,
    neighbour_thres=0.1,
):
    """Estimate the second-order ALE function for two continuous feature data.

    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which the model was trained.
    features : 2-iterable of column label
        The two desired features, as two column labels.
    bins : [2-iterable of] int
        This defines the number of bins to compute. The effective number of bins may
        be less than this as only unique quantile values of train_set[feature] are
        used. If one integer is given, this is used for both features.
    include_mean : bool, optional
        Add the mean of all predictions to the output. Useful for evaluating joint
        effects.
    n_jobs : int, optional
        The number of cores to use for parallel processing when querying for nearest
        neighbour bins while substituting empty bins. If -1 is given, all processors
        are used.
    n_neighbour : int, optional
        Number of nearest neighbour bins to average when substituting empty bins.
    neighbour_thres : float in [0, 1], optional
        Limit the number of nearest neighbour bins used to substitute empty bins to at
        most `n_neighbour` or the number of cells needed to account for a fraction of
        `neighbour_thres` of samples, which ever is smaller.

    Returns
    -------
    quantiles : 2-tuple of array
        The quantiles used: first the quantiles for `features[0]` with shape (M + 1,),
        then for `features[1]` with shape (N + 1,).
    ale : (M + 1, N + 1) masked array
        The second order ALE. Elements are masked where all adjacent quantiles touch
        cells which did not contain any data (see `samples`).
    samples : (M, N) array
        The number of samples in each quantile bin.

    Raises
    ------
    ValueError
        If `features` does not contain 2 features.
    ValueError
        If more than 2 bins are given.
    ValueError
        If bins are not integers.

    """
    features = _parse_features(features)
    if len(features) != 2:
        raise ValueError(
            "'features' contained '{n_feat}' features. Expected 2.".format(
                n_feat=len(features)
            )
        )

    quantiles_list, bins_list = tuple(
        zip(
            *(
                _get_quantiles(train_set, feature, n_bin)
                for feature, n_bin in zip(features, _check_two_ints(bins))
            )
        )
    )
    logger.debug("Quantiles: {}.", quantiles_list)

    # Define the bins the feature samples fall into. Shift and clip to ensure we are
    # getting the index of the left bin edge and the smallest sample retains its index
    # of 0.
    indices_list = [
        np.clip(np.digitize(train_set[feature], quantiles, right=True) - 1, 0, None)
        for feature, quantiles in zip(features, quantiles_list)
    ]

    # Invoke the predictor at the corners of the bins. Then compute the second order
    # difference between the predictions at the bin corners.
    predictions = {}
    for shifts in product(*(range(2),) * 2):
        mod_train_set = train_set.copy()
        for i in range(2):
            mod_train_set[features[i]] = quantiles_list[i][indices_list[i] + shifts[i]]
        predictions[shifts] = predictor(mod_train_set)

    # The individual effects.
    effects = (predictions[(1, 1)] - predictions[(1, 0)]) - (
        predictions[(0, 1)] - predictions[(0, 0)]
    )

    # Group the effects by their indices along both axes.
    index_groupby = pd.DataFrame(
        {"index_0": indices_list[0], "index_1": indices_list[1], "effects": effects}
    ).groupby(["index_0", "index_1"])

    # Compute mean effects.
    mean_effects = index_groupby.mean()
    # Get the indices of the mean values.
    group_indices = mean_effects.index
    valid_grid_indices = tuple(zip(*group_indices))
    # Extract only the data.
    mean_effects = mean_effects.to_numpy().flatten()

    # Get the number of samples in each bin.
    n_samples = index_groupby.size().to_numpy()

    # Create a 2D array of the number of samples in each bin.
    samples = np.zeros(bins_list)
    samples[valid_grid_indices] = n_samples

    ale = np.zeros((len(quantiles_list[0]), len(quantiles_list[1])))

    # Place the mean effects into the final array.
    # The first row and column are effectively filled with 0s.
    ale[1:, 1:][valid_grid_indices] = mean_effects

    # Record where elements were missing.
    missing_bin_mask = np.ones(bins_list, dtype=np.bool_)
    missing_bin_mask[valid_grid_indices] = False

    if np.any(missing_bin_mask):
        # Replace missing entries with their nearest neighbours.

        # Calculate the dense location matrices (for both features) of all bin centres.
        centres_list = np.meshgrid(
            *(_get_centres(quantiles) for quantiles in quantiles_list), indexing="ij"
        )

        # Select only those bin centres which are valid (had observation).
        valid_indices = np.where(~missing_bin_mask)
        tree = cKDTree(
            np.hstack(
                tuple(centres[valid_indices][:, np.newaxis] for centres in centres_list)
            )
        )

        # There can only be as many nearest neighbours as there are valid bins.
        n_neighbour = min((n_neighbour, (~missing_bin_mask).sum()))

        # Determine the indices of the points which are nearest to the empty bins.
        distances, nearest_points = tree.query(
            np.hstack(
                tuple(
                    centres[missing_bin_mask][:, np.newaxis] for centres in centres_list
                )
            ),
            k=n_neighbour,
            n_jobs=n_jobs,
        )

        # Go from the indices into `tree.data` returned by `query()` above to indices
        # into the ALE array (and equivalently, the number of samples array).
        nearest_indices = tuple(
            axis_indices[nearest_points] for axis_indices in valid_indices
        )

        to_average = ale[1:, 1:][nearest_indices]

        # Limit the number of averaged nearest neighbours using `neighbour_thres`.
        # Substitute `np.nan` for bins that need to be excluded so as not to exceed
        # `neighbour_thres`. Averaging using `np.nanmean()` then excludes these.
        threshold_samples = math.ceil(neighbour_thres * train_set.shape[0])
        n_avg_samples = np.cumsum(samples[nearest_indices], axis=1)

        exclude_mask = n_avg_samples > threshold_samples

        to_average[exclude_mask] = np.nan

        # Replace the invalid bin values with averages over the nearest neighbours.
        ale[1:, 1:][missing_bin_mask] = np.nanmean(to_average, axis=1)

    # Compute the cumulative sums.
    ale = np.cumsum(np.cumsum(ale, axis=0), axis=1)

    # Subtract first order effects along both axes separately.
    for i in range(2):
        # Depending on `i`, reverse the arguments to operate on the opposite axis.
        flip = slice(None, None, 1 - 2 * i)

        # Undo the cumulative sum along the axis.
        first_order = ale[(slice(1, None), ...)[flip]] - ale[(slice(-1), ...)[flip]]
        # Average the diffs across the other axis.
        first_order = (
            first_order[(..., slice(1, None))[flip]]
            + first_order[(..., slice(-1))[flip]]
        ) / 2
        # Weight by the number of samples in each bin.
        first_order *= samples
        # Take the sum along the axis.
        first_order = np.sum(first_order, axis=1 - i)
        # Normalise by the number of samples in the bins along the axis.
        first_order /= np.sum(samples, axis=1 - i)
        # The final result is the cumulative sum (with an additional 0).
        first_order = np.array([0, *np.cumsum(first_order)]).reshape((-1, 1)[flip])

        # Subtract the first order effect.
        ale -= first_order

    # Centre the ALE by subtracting its expectation value.
    ale -= np.sum(_get_centres(ale, axis=(0, 1)) * samples) / train_set.shape[0]

    if include_mean:
        ale += sum(np.mean(preds) for preds in predictions.values()) / 4

    # Compute the ALE mask.
    pad_missing = np.ones(np.array(missing_bin_mask.shape) + 2, dtype=np.bool_)
    pad_missing[1:-1, 1:-1] = missing_bin_mask
    pad_missing_counts = _get_centres(pad_missing, axis=(0, 1))

    # Adjacent quantiles touching only empty cells yield average counts of 1.
    masked_ale = np.ma.MaskedArray(ale, mask=np.isclose(pad_missing_counts, 1))

    return quantiles_list, masked_ale, samples


def first_order_ale_cat(
    predictor, train_set, feature, features_classes, feature_encoder=None
):
    """Compute the first-order ALE function on single categorical feature data.

    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which model was trained.
    feature : str
        Feature name.
    features_classes : iterable or str
        Feature's classes.
    feature_encoder : callable or iterable
        Encoder that was used to encode categorical feature. If features_classes is
        not None, this parameter is skipped.

    """
    num_cat = len(features_classes)
    ale = np.zeros(num_cat)  # Final ALE function.

    for i in range(num_cat):
        subset = train_set[train_set[feature] == features_classes[i]]

        # Without any observation, local effect on split area is null.
        if len(subset) != 0:
            z_low = subset.copy()
            z_up = subset.copy()
            # The main ALE idea that compute prediction difference between same data
            # except feature's one.
            z_low[feature] = quantiles[i - 1]
            z_up[feature] = quantiles[i]
            ale[i] += (predictor(z_up) - predictor(z_low)).sum() / subset.shape[0]

    # The accumulated effect.
    ale = ale.cumsum()
    # Now we have to center ALE function in order to obtain null expectation for ALE
    # function.
    ale -= ale.mean()
    return ale


def _compute_mc_hull_poly_points(mc_data, interp_quantiles, verbose=False):
    """Compute the convex hull polygon points.

    Parameters
    ----------
    mc_data : iterable of tuple
        The quantiles and ALE data for the Monte-Carlo repetitions.
    interp_quantiles : array
        The quantiles for which to compute the polygon points.

    Returns
    -------
    points : array
        The polygon points.

    """
    # Store the points for each new quantile.
    interpolated_data = np.ma.MaskedArray(
        np.zeros((len(mc_data), interp_quantiles.size)),
        mask=True,  # Only treat those points as valid which are assigned later on.
    )

    for (mc_index, (mc_quantiles, mc_ale)) in enumerate(
        tqdm(mc_data, desc="MC hull Polygon", disable=not verbose)
    ):
        # Do not extrapolate.
        valid_mask = (interp_quantiles >= mc_quantiles[0]) & (
            interp_quantiles <= mc_quantiles[-1]
        )
        interpolated = np.interp(interp_quantiles[valid_mask], mc_quantiles, mc_ale)
        # Save the interpolated data at the new quantiles.
        interpolated_data[mc_index][valid_mask] = interpolated

    # The Polygon will consist of the mean +- the std.
    interp_means = np.mean(interpolated_data, axis=0)
    interp_std = np.std(interpolated_data, axis=0)

    points = np.empty((interp_quantiles.size * 2, 2))
    points[: interp_quantiles.size, 0] = interp_quantiles
    points[: interp_quantiles.size, 1] = interp_means + interp_std
    points[interp_quantiles.size :, 0] = interp_quantiles[::-1]
    points[interp_quantiles.size :, 1] = (interp_means - interp_std)[::-1]

    # Ignore those points for which no non-interpolated samples could be found.
    points = points[
        ~np.any(np.isclose(np.abs(points), np.finfo(np.float64).max), axis=1)
    ]

    return points


def _mc_replicas(
    model,
    train_set,
    train_response,
    monte_carlo_ratio,
    monte_carlo_rep,
    bins,
    quantiles,
    mod_quantiles,
    features,
    rng,
    quantile_axis=False,
    center=False,
    ale=None,
    verbose=False,
):
    if isinstance(monte_carlo_ratio, (int, np.integer)):
        rep_size = monte_carlo_ratio
    else:
        rep_size = int(monte_carlo_ratio * train_set.shape[0])

    mc_replica_data = []

    for _ in tqdm(
        range(monte_carlo_rep),
        desc="Calculating MC replicas",
        disable=not verbose,
    ):
        # Bootstrap sampling (with replacement).
        bootstrap_inds = rng.integers(low=0, high=train_set.shape[0], size=rep_size)
        train_set_rep = train_set.iloc[bootstrap_inds]
        train_response_rep = train_response[bootstrap_inds]

        # Copy and refit the model.
        mc_model = clone(model, safe=False)
        mc_model.fit(train_set_rep, train_response_rep)

        # The same quantiles cannot be reused here as this could cause
        # some bins to be empty or contain disproportionate numbers of
        # samples.
        mc_quantiles, mc_ale = first_order_ale_quant(
            mc_model.predict, train_set_rep, features[0], bins
        )
        if center:
            # Align start of ALE plots to the overall ALE plot.
            mc_ale -= mc_ale[0] - ale[0]
        if quantile_axis:
            # Interpolate the quantiles to the original quantiles.
            mc_quantiles = np.interp(mc_quantiles, quantiles, mod_quantiles)

        mc_replica_data.append((mc_quantiles, mc_ale))

    return mc_replica_data


def ale_plot(
    model,
    train_set,
    features,
    bins=10,
    train_response=None,
    fig=None,
    ax=None,
    monte_carlo=False,
    features_classes=None,
    monte_carlo_rep=50,
    monte_carlo_ratio=0.1,
    monte_carlo_hull=False,
    monte_carlo_hull_points=300,
    rugplot_lim=1000,
    verbose=False,
    plot_quantiles=False,
    center=False,
    quantile_axis=False,
    scilim=2,
    include_first_order=False,
    plot_kwargs=None,
    grid_kwargs=None,
    hull_polygon_kwargs=None,
    n_jobs=1,
    n_neighbour=10,
    neighbour_thres=0.1,
    return_data=False,
    return_mc_data=False,
    rng=None,
):
    """Plots ALE function of specified features based on training set.

    Parameters
    ----------
    model : object
        An object that implements a 'predict' method. Additionally, it should also
        implement a 'fit' method if Monte-Carlo replicas will be computed (in which
        case the model will be copied using 'sklearn.base.clone' prior to being fitted with
        the bootstrap samples.
    train_set : pandas.core.frame.DataFrame
        Training set predictors on which model was trained.
    features : [2-iterable of] column label
        One or two features for which to plot the ALE plot.
    bins : [2-iterable of] int, optional
        Number of bins used to split feature's space. 2 integers can only be given
        when 2 features are supplied in order to compute a different number of
        quantiles for each feature.
    train_response : array-like or None
        Training set response. Only needs to be given if `monte_carlo` is True.
    fig : matplotlib Figure
        Figure to plot onto. Required to plot colorbar for 2D ALE plots. The current
        figure and axes will be used if both `fig` and `ax` are None (default). Will
        be used to create `ax` if only `ax` is `None`.
    ax : matplotlib Axes
        Axes to plot onto. If None (default), the current axis will be used (a new one
        will be created if needed).
    monte_carlo : boolean, optional
        Compute and plot Monte-Carlo samples.
    features_classes : iterable of str, optional
        If features is first-order and a categorical variable, plot ALE according to
        discrete aspect of data.
    monte_carlo_rep : int
        Number of Monte-Carlo replicas.
    monte_carlo_ratio : float or int
        If a float is given, this determines the proportion of randomly selected
        samples from `train_set` for each Monte-Carlo replica. An integer sets the
        number of samples directly.
    monte_carlo_hull : bool
        If True, plot the concave hull of the Monte-Carlo replicas instead of the
        individual curves.
    monte_carlo_hull_points : int
        An even number of points to use for the construction of the hull polygon.
    rugplot_lim : int, optional
        If `train_set` has more rows than `rugplot_lim`, no rug plot will be plotted.
        Set to None to always plot rug plots. Set to 0 to disable rug plots.
    verbose : bool, optional
        If True, output additional information, such as Monte Carlo progress updates.
    plot_quantiles : {False, True, 'x', 'y', 'both', ('x', 'y')}, optional
        Show the feature quantiles (via added axes). Axes can be selected using 'x',
        'y', 'both', or ('x', 'y').
    center : bool, optional
        Only applies to first-order ALE plotting (1 feature). If True, align the
        initial points of the Monte Carlo replicas to more easily isolate their
        divergence from the overall first-order ALE. The resulting lines will no
        longer represent true ALE plots.
    quantile_axis : {False, True, 'x', 'y', 'both', ('x', 'y')}, optional
        If True, quantiles are evenly spaced (along both axes for a 2D ALE plot). Axes
        can be selected using 'x', 'y', 'both', or ('x', 'y').
    scilim : float, optional
        If the decimal logarithm of the axis tick labels varies by more than `scilim`,
        tick labels will be formatted using scientific notation.
    include_first_order : bool, optional
        If two features are given, this decides whether to add first order ALEs and
        mean to the plot to visualise the combined effects.
    plot_kwargs : dict, optional
        Parameters that are passed to `kwargs` for `first_order_*_plot` or
        `second_order_*_plot` for one or two features, respectively. See
        'Other Parameters' for these functions for more details. For the Monte Carlo
        plots, parameters can be given using the 'mc_' prefix, e.g. 'mc_color' to
        determine the color of the Monte Carlo lines.
    grid_kwargs : bool or dict, optional
        Parameters passed to `ax.grid()` for a 1D ALE plot. Giving False disables the
        plotting of a grid. By default, major grid lines are shown with a linestyle of
        '--' and alpha value of 0.4.
    hull_polygon_kwargs : dict or None, optional
        Additional arguments to pass to the matplotlib.patches.Polygon constructor
        (e.g. 'facecolor' or 'alpha').
    n_jobs : int, optional
        Applies to 2D ALE plotting only. The number of cores to use for parallel
        processing when querying for nearest neighbour bins while substituting empty
        bins. If -1 is given, all processors are used.
    n_neighbour : int, optional
        Applies to 2D ALE plotting only. Number of nearest neighbour bins to average
        when substituting empty bins.
    neighbour_thres : float in [0, 1], optional
        Applies to 2D ALE plotting only. Limit the number of nearest neighbour bins
        used to substitute empty bins to at most `n_neighbour` or the number of cells
        needed to account for a fraction of `neighbour_thres` of samples, which ever
        is smaller.
    return_data : bool, optional
        Return the output of `first_order_ale_quant()` for first-order ALE plots, and
        the output of `second_order_ale_quant()` for second-order ALE plots.
    return_mc_data : bool, optional
        Return the Monte Carlo quantile and ALE data for the first order ALE plot.
    rng : numpy.random.Generator
        Can be given to draw identical samples for Monte-Carlo replicas.

    Returns
    -------
    fig : matplotlib Figure
        ALE plot figure.
    axes : dict of matplotlib Axes
        Axes containing the ALE plot and other elements like a colorbar or quantile
        axes, depending on the parameters given.
    data : tuple
        Present only if `return_data` is `True`. Contains the return values of either
        `first_order_ale_quant()` or `second_order_ale_quant()` depending on the
        number of items in `features`.
    mc_data : tuple
        Present only if `return_mc_data` is `True`. Contains the return values of
        `first_order_ale_quant()` for the Monte Carlo replicas. If a second-order ALE
        plot is carried out or `monte_carlo` is `False`, this will be empty.

    Raises
    ------
    ValueError
        If `fig` is `None`, but `ax` is not None.
    ValueError
        If `len(features)` not in {1, 2}.
    ValueError
        If multiple bins were given for 1 feature.
    ValueError
        If `monte_carlo` is True, but `train_response` is None.
    NotImplementedError
        If `features_classes` is not None.

    """
    features = _parse_features(features)

    if monte_carlo:
        if train_response is None:
            raise ValueError(
                "If `monte_carlo` is True, `train_response` needs to be given."
            )
        else:
            train_response = np.asarray(train_response)

    if fig is None and ax is None:
        logger.debug("Getting current figure and axis.")
        fig, ax = plt.gcf(), plt.gca()
    elif fig is not None and ax is None:
        logger.debug("Creating axis from figure {}.", fig)
        ax = fig.add_subplot(111)
    elif fig is None and ax is not None and len(features) == 2:
        raise ValueError("An axis ('ax') was supplied without a figure ('fig').")

    axes = {"ale": ax}
    return_vals = [fig, axes]
    mc_return_vals = []

    if features_classes is not None:
        raise NotImplementedError("'features_classes' is not implemented yet.")

    if plot_kwargs is None:
        plot_kwargs = {}

    mc_plot_kwargs = plot_kwargs.copy()

    # Override using Monte-Carlo specific options and remove these from the main
    # plot_kwargs.
    given_mc_options = []
    for name in {name for name in plot_kwargs if name.startswith("mc_")}:
        given_mc_options.append(name)
        mc_plot_kwargs[name.lstrip("mc_")] = plot_kwargs.pop(name)
        del mc_plot_kwargs[name]

    if "mc_color" not in given_mc_options:
        mc_plot_kwargs["color"] = "#1f77b4"
    if "mc_alpha" not in given_mc_options:
        mc_plot_kwargs["alpha"] = 0.06

    if hull_polygon_kwargs is None:
        hull_polygon_kwargs = {}

    ax_labels = ["Feature '{}'".format(feature) for feature in features]
    predictor = model.predict

    if len(features) == 1:
        if plot_quantiles:
            if plot_quantiles in ("y", "both"):
                raise ValueError(
                    "If one feature is given, only 'x' or True are supported "
                    f"for 'plot_quantiles'. Got: {repr(plot_quantiles)}."
                )
        if quantile_axis:
            if quantile_axis in ("y", "both"):
                raise ValueError(
                    "If one feature is given, only 'x' or True are supported "
                    f"for 'quantile_axis'. Got: {repr(quantile_axis)}."
                )
        if not isinstance(bins, (int, np.integer)):
            raise ValueError("1 feature was given, but 'bins' was not an integer.")

        if rng is None:
            rng = np.random.default_rng()

        if features_classes is None:
            # Continuous data.
            quantiles, ale = first_order_ale_quant(
                predictor, train_set, features[0], bins
            )
            if return_data:
                return_vals.append((quantiles, ale))
            if quantile_axis:
                mod_quantiles = np.arange(len(quantiles))
                ax.set_xticks(mod_quantiles)
                ax.set_xticklabels(_sci_format(quantiles, scilim=scilim))
            else:
                mod_quantiles = quantiles

            if monte_carlo:
                mc_replica_data = _mc_replicas(
                    model=clone(model, safe=False),
                    train_set=train_set,
                    train_response=train_response,
                    monte_carlo_ratio=monte_carlo_ratio,
                    monte_carlo_rep=monte_carlo_rep,
                    bins=bins,
                    quantiles=quantiles,
                    mod_quantiles=mod_quantiles,
                    features=features,
                    rng=rng,
                    quantile_axis=quantile_axis,
                    center=center,
                    ale=ale,
                    verbose=verbose,
                )
                if not monte_carlo_hull:
                    # Plot individual lines immediately instead of plotting the hull
                    # later.
                    for mc_quantiles, mc_ale in mc_replica_data:
                        first_order_quant_plot(
                            mc_quantiles, mc_ale, ax=ax, **mc_plot_kwargs
                        )

                if return_mc_data or monte_carlo_hull:
                    # Need to record the individual runs.
                    mc_return_vals.extend(mc_replica_data)

                if return_mc_data:
                    return_vals.append(tuple(mc_return_vals))

                if monte_carlo_hull:
                    if "facecolor" not in hull_polygon_kwargs:
                        hull_polygon_kwargs["facecolor"] = "C0"
                    if "alpha" not in hull_polygon_kwargs:
                        hull_polygon_kwargs["alpha"] = 0.2

                    # Compute the hull and plot it as a Polygon.
                    ax.add_patch(
                        Polygon(
                            _compute_mc_hull_poly_points(
                                mc_return_vals,
                                np.linspace(
                                    np.min(
                                        [
                                            mc_quantiles[0]
                                            for mc_quantiles, mc_ale in mc_replica_data
                                        ]
                                    ),
                                    np.max(
                                        [
                                            mc_quantiles[-1]
                                            for mc_quantiles, mc_ale in mc_replica_data
                                        ]
                                    ),
                                    monte_carlo_hull_points // 2,
                                ),
                                verbose=verbose,
                            ),
                            **hull_polygon_kwargs,
                        )
                    )

            _ax_labels(ax, *ax_labels)
            mc_string = monte_carlo_rep if monte_carlo else "False"
            _ax_title(
                ax,
                f"First-order ALE of feature '{features[0]}'",
                f"Bins : {len(quantiles) - 1} - Monte-Carlo : {mc_string}",
            )

            if grid_kwargs is None:
                grid_kwargs = {}

            if "linestyle" not in grid_kwargs:
                grid_kwargs["linestyle"] = "--"
            if "alpha" not in grid_kwargs:
                grid_kwargs["alpha"] = 0.4

            if grid_kwargs:
                ax.grid(**grid_kwargs)

            if rugplot_lim is None or train_set.shape[0] <= rugplot_lim:
                sns.rugplot(train_set[features[0]], ax=ax, alpha=0.2)

            if "color" not in plot_kwargs:
                plot_kwargs["color"] = "black"

            first_order_quant_plot(mod_quantiles, ale, ax=ax, **plot_kwargs)

            if plot_quantiles:
                axes["quantiles_x"] = _ax_quantiles(ax, mod_quantiles)

    elif len(features) == 2:
        if plot_quantiles in ("x", "y"):
            plot_quantiles = (plot_quantiles,)
        elif plot_quantiles:
            plot_quantiles = ("x", "y")
        else:
            plot_quantiles = ()

        if quantile_axis in ("x", "y"):
            quantile_axis = (quantile_axis,)
        elif quantile_axis:
            quantile_axis = ("x", "y")
        else:
            quantile_axis = ()

        if features_classes is None:
            # Continuous data.

            # Verify the bins and make sure two ints are stored in `bins`.
            bins = _check_two_ints(bins)

            quantiles_list, ale, samples_grid = second_order_ale_quant(
                predictor,
                train_set,
                features,
                bins,
                include_mean=include_first_order,
                n_jobs=n_jobs,
                n_neighbour=n_neighbour,
                neighbour_thres=neighbour_thres,
            )
            if return_data:
                return_vals.append((quantiles_list, ale, samples_grid))

            if include_first_order:
                # Compute first order ALEs.
                for i, (feature, nbins) in enumerate(zip(features, bins)):
                    _, first_ord_ale = first_order_ale_quant(
                        predictor, train_set, feature, nbins
                    )
                    # Add the first order ALE to the second order ALE.
                    new_shape = [1, 1]
                    new_shape[i] = -1
                    ale += first_ord_ale.reshape(*new_shape)
                title = f"Combined ALE of features '{features[0]}' and '{features[1]}'"

            else:
                # Update plotting kwargs to match the type of requested visualisation.
                if "colorbar" not in plot_kwargs:
                    plot_kwargs["colorbar"] = "symmetric"
                title = (
                    f"Second-order ALE of features '{features[0]}' and '{features[1]}'"
                )

            if quantile_axis:
                mod_quantiles_list = []
                for axis, quantiles in zip(("x", "y"), quantiles_list):
                    if axis in quantile_axis:
                        inds = np.arange(len(quantiles))
                        mod_quantiles_list.append(inds)
                        ax.set(**{f"{axis}ticks": inds})
                        ax.set(
                            **{
                                f"{axis}ticklabels": _sci_format(
                                    quantiles, scilim=scilim
                                )
                            }
                        )
                    else:
                        mod_quantiles_list.append(quantiles)
            else:
                mod_quantiles_list = quantiles_list

            if plot_quantiles:
                for twin, quantiles in zip(("x", "y"), mod_quantiles_list):
                    if twin in plot_quantiles:
                        axes[f"quantiles_{twin}"] = _ax_quantiles(
                            ax, quantiles, twin=twin
                        )

                if "x" in plot_quantiles:
                    if "colorbar_kwargs" not in plot_kwargs:
                        plot_kwargs["colorbar_kwargs"] = {}

                    if "pad" not in plot_kwargs["colorbar_kwargs"]:
                        plot_kwargs["colorbar_kwargs"]["pad"] = 0.1

            axes.update(
                second_order_quant_plot(
                    mod_quantiles_list, ale, samples_grid, fig=fig, ax=ax, **plot_kwargs
                )[1]
            )

            _ax_labels(ax, *ax_labels)
            resultant_bins = [len(quant) - 1 for quant in quantiles_list]
            _ax_title(ax, title, "Bins : {0}x{1}".format(*resultant_bins))
    else:
        raise ValueError(
            f"'{len(features)}' 'features' were given, but only 1 or 2 are supported."
        )
    return tuple(return_vals)
