# -*- coding: utf-8 -*-
"""ALE plotting.

TODO:
    Clarify some docstrings.
    Do not set defaults for matplotlib colors, linestyles, etc... in the code - leave
    this up to the user, use matplotlib styles, etc...

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from loguru import logger


def _ax_title(ax, title, subtitle):
    """Add title to axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add title to.
    title : str
        Axis title.
    subtitle : str
        Sub-title for figure.

    """
    ax.set_title("\n".join((title, subtitle)))


def _ax_labels(ax, xlabel, ylabel):
    """Add labels to axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add labels to.
    xlabel : str
        X axis label.
    ylabel : str
        Y axis label.

    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _ax_quantiles(ax, quantiles, twin="x"):
    """Plot quantiles of a feature over opposite axis.

    Parameters
    ----------
    ax : matplotlib.Axis
        Axis to work with.
    quantiles : array-like
        Quantiles to plot.
    twin : {'x', 'y'}
        Possible values are 'x' or 'y', depending on which axis to plot quantiles onto.

    Raises
    ------
    ValueError
        If `twin` is not one of 'x' or 'y'.

    """
    # XXX: Clarify docs - are quantiles plotted 'onto' the axis, or the opposite axis?
    if twin not in ("x", "y"):
        raise ValueError("'twin' should be one of 'x' or 'y'.")

    logger.debug("Quantiles : {}.", quantiles)

    ax_mod = ax.twiny() if twin == "x" else ax.twinx()
    getattr(ax_mod, f"set_{twin}ticks")(quantiles)
    getattr(ax_mod, f"set_{twin}ticklabels")(
        [
            "{1:0.{0}f}%".format(
                int(i / (len(quantiles) - 1) * 100 % 1 > 0),
                i / (len(quantiles) - 1) * 100,
            )
            for i in range(len(quantiles))
        ],
        color="#545454",
        fontsize=7,
    )
    getattr(ax_mod, f"set_{twin}lim")(getattr(ax, f"get_{twin}lim")())


def _first_order_quant_plot(ax, quantiles, ALE, **kwargs):
    ax.plot((quantiles[1:] + quantiles[:-1]) / 2, ALE, **kwargs)


def _second_order_quant_plot(fig, ax, quantiles, ALE, **kwargs):
    logger.debug(ALE)
    x = np.linspace(quantiles[0][0], quantiles[0][-1], 50)
    y = np.linspace(quantiles[1][0], quantiles[1][-1], 50)
    X, Y = np.meshgrid(x, y)
    ALE_interp = scipy.interpolate.interp2d(quantiles[0], quantiles[1], ALE)
    CF = ax.contourf(X, Y, ALE_interp(x, y), cmap="bwr", levels=30, alpha=0.7)
    fig.colorbar(CF)


def _first_order_ale_quant(predictor, train_set, feature, quantiles):
    """Estimate the first-order ALE function on single continuous feature data.

    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which the model was trained.
    feature : str
        Feature name.
    quantiles : array-like
        Feature quantiles.

    """
    ALE = np.zeros(len(quantiles) - 1)  # Final ALE function

    for i in range(1, len(quantiles)):
        subset = train_set[
            (quantiles[i - 1] <= train_set[feature])
            & (train_set[feature] < quantiles[i])
        ]

        # Without any observation, local effect on splitted area is null
        if len(subset) != 0:
            z_low = subset.copy()
            z_up = subset.copy()
            # The main ALE idea that compute prediction difference between same data
            # except feature's one
            z_low[feature] = quantiles[i - 1]
            z_up[feature] = quantiles[i]
            ALE[i - 1] += (predictor(z_up) - predictor(z_low)).sum() / subset.shape[0]

    # The accumulated effect
    ALE = ALE.cumsum()
    # Now we have to center ALE function in order to obtain null expectation for ALE
    # function
    ALE -= ALE.mean()
    return ALE


def _second_order_ale_quant(predictor, train_set, features, quantiles):
    """Computes second-order ALE function on two continuous features data.

    """
    quantiles = np.asarray(quantiles)
    ALE = np.zeros((quantiles.shape[1], quantiles.shape[1]))  # Final ALE function
    print(quantiles)

    for i in range(1, len(quantiles[0])):
        for j in range(1, len(quantiles[1])):
            # Select subset of training data that falls within subset
            subset = train_set[
                (quantiles[0, i - 1] <= train_set[features[0]])
                & (quantiles[0, i] > train_set[features[0]])
                & (quantiles[1, j - 1] <= train_set[features[1]])
                & (quantiles[1, j] > train_set[features[1]])
            ]

            # Without any observation, local effect on splitted area is null
            if len(subset) != 0:
                z_low = [
                    subset.copy() for _ in range(2)
                ]  # The lower bounds on accumulated grid
                z_up = [
                    subset.copy() for _ in range(2)
                ]  # The upper bound on accumulated grid
                # The main ALE idea that compute prediction difference between same
                # data except feature's one
                z_low[0][features[0]] = quantiles[0, i - 1]
                z_low[0][features[1]] = quantiles[1, j - 1]
                z_low[1][features[0]] = quantiles[0, i]
                z_low[1][features[1]] = quantiles[1, j - 1]
                z_up[0][features[0]] = quantiles[0, i - 1]
                z_up[0][features[1]] = quantiles[1, j]
                z_up[1][features[0]] = quantiles[0, i]
                z_up[1][features[1]] = quantiles[1, j]

                ALE[i, j] += (
                    predictor(z_up[1])
                    - predictor(z_up[0])
                    - (predictor(z_low[1]) - predictor(z_low[0]))
                ).sum() / subset.shape[0]

    ALE = np.cumsum(ALE, axis=0)  # The accumulated effect
    # Now we have to center ALE function in order to obtain null expectation for ALE
    # function
    ALE -= ALE.mean()
    return ALE


def _first_order_ale_cat(
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
    ALE = np.zeros(num_cat)  # Final ALE function

    for i in range(num_cat):
        subset = train_set[train_set[feature] == features_classes[i]]

        # Without any observation, local effect on splitted area is null
        if len(subset) != 0:
            z_low = subset.copy()
            z_up = subset.copy()
            # The main ALE idea that compute prediction difference between same data except feature's one
            z_low[feature] = quantiles[i - 1]
            z_up[feature] = quantiles[i]
            ALE[i] += (predictor(z_up) - predictor(z_low)).sum() / subset.shape[0]

    # The accumulated effect
    ALE = ALE.cumsum()
    # Now we have to center ALE function in order to obtain null expectation for ALE
    # function
    ALE -= ALE.mean()
    return ALE


def ale_plot(
    model,
    train_set,
    features,
    bins=10,
    monte_carlo=False,
    predictor=None,
    features_classes=None,
    **kwargs,
):
    """Plots ALE function of specified features based on training set.

    Parameters
    ----------
    model : object
        An object that implements a 'predict' method. If None, a `predictor` function
        must be supplied which will be used instead of `model.predict`.
    train_set : pandas.core.frame.DataFrame
        Training set on which model was trained.
    features : str or tuple of str
        One or two features for which to plot ALE plot.
    bins : int, optional
        Number of bins used to split feature's space.
    monte_carlo : boolean, optional
        Compute and plot Monte-Carlo samples.
    predictor : callable
        Custom prediction function. See `model`.
    features_classes : iterable of str, optional
        If features is first-order and a categorical variable, plot ALE according to
        discrete aspect of data. Note: not implemented yet.
    monte_carlo_rep : int
        Number of Monte-Carlo replicas.
    monte_carlo_ratio : float
        Proportion of randomly selected samples from dataset for each Monte-Carlo
        replica.

    Raises
    ------
    ValueError
        If both `model` and `predictor` are None.

    """
    if model is None and predictor is None:
        raise ValueError("If 'model' is None, 'predictor' must be supplied.")

    if features_classes is not None:
        raise NotImplementedError("'features_classes' is not implemented yet.")

    fig, ax = plt.subplots()

    if not isinstance(features, (list, tuple, np.ndarray)):
        features = np.asarray([features])

    if len(features) == 1:
        quantiles = np.percentile(
            train_set[features[0]], [1.0 / bins * i * 100 for i in range(0, bins + 1)]
        )  # Split areas of feature

        if monte_carlo:
            mc_rep = kwargs.get("monte_carlo_rep", 50)
            mc_ratio = kwargs.get("monte_carlo_ratio", 0.1)
            mc_replicates = np.asarray(
                [
                    [
                        np.random.choice(range(train_set.shape[0]))
                        for _ in range(int(mc_ratio * train_set.shape[0]))
                    ]
                    for _ in range(mc_rep)
                ]
            )
            for k, rep in enumerate(mc_replicates):
                train_set_rep = train_set.iloc[rep, :]
                if features_classes is None:
                    mc_ALE = _first_order_ale_quant(
                        model.predict if predictor is None else predictor,
                        train_set_rep,
                        features[0],
                        quantiles,
                    )
                    _first_order_quant_plot(
                        ax, quantiles, mc_ALE, color="#1f77b4", alpha=0.06
                    )

        if features_classes is None:
            ALE = _first_order_ale_quant(
                model.predict if predictor is None else predictor,
                train_set,
                features[0],
                quantiles,
            )
            _ax_labels(ax, "Feature '{}'".format(features[0]), "")
            _ax_title(
                ax,
                "First-order ALE of feature '{0}'".format(features[0]),
                "Bins : {0} - Monte-Carlo : {1}".format(
                    len(quantiles) - 1,
                    mc_replicates.shape[0] if monte_carlo else "False",
                ),
            )
            ax.grid(True, linestyle="-", alpha=0.4)
            sns.rugplot(train_set[features[0]], ax=ax, alpha=0.2)
            _first_order_quant_plot(ax, quantiles, ALE, color="black")
            _ax_quantiles(ax, quantiles)
    elif len(features) == 2:
        quantiles = [
            np.percentile(
                train_set[f], [1.0 / bins * i * 100 for i in range(0, bins + 1)]
            )
            for f in features
        ]

        if features_classes is None:
            ALE = _second_order_ale_quant(
                model.predict if predictor is None else predictor,
                train_set,
                features,
                quantiles,
            )
            _second_order_quant_plot(fig, ax, quantiles, ALE)
            _ax_labels(
                ax,
                "Feature '{}'".format(features[0]),
                "Feature '{}'".format(features[1]),
            )
            _ax_quantiles(ax, quantiles[0], twin="x")
            _ax_quantiles(ax, quantiles[1], twin="y")
            _ax_title(
                ax,
                "Second-order ALE of features '{0}' and '{1}'".format(
                    features[0], features[1]
                ),
                "Bins : {0}x{1}".format(len(quantiles[0]) - 1, len(quantiles[1]) - 1),
            )

    plt.show()
