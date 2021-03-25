# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from alphashape import alphashape
from descartes import PolygonPatch
from tqdm.auto import tqdm

from alepython.ale import _ax_title, _sci_format, ale_plot

__all__ = ("multi_ale_plot_1d",)


def multi_ale_plot_1d(
    features,
    title=None,
    xlabel=None,
    ylabel=None,
    x_rotation=20,
    markers=("o", "v", "^", "<", ">", "x", "+"),
    colors=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    zorders=None,
    xlabel_skip=2,
    format_xlabels=True,
    **kwargs,
):
    """Plots ALE function of multiple specified features based on training set.

    Multiple first-order (1D) ALE plots will be computed and plotted on the same plot.

    Note that currently, only concave hull plotting of Monte-Carlo replicas is
    supported.

    Parameters
    ----------
    features : iterable of column label
        Features for which to plot the 1D ALE plot.
    title : str or None
        Figure title.
    xlabel : str or None
        Figure x-label.
    ylabel : str or None
        Figure y-label.
    x_rotation : x-label rotation.
    markers : iterable of str
        Matplotlib markers used to differentiate the different features.
    colors : iterable
        Matplotlib colors used to differentiate the different features.
    zorders : iterable of int or None
        zorder used for each feature, with the hull (if applicable) having the same
        zorder as the ALE line plot. By default, the last feature will have the
        highest, and the first feature the lowest zorder.
    xlabel_skip : int
        Only plot an xlabel marker every `xlabel_skip` label.
    format_xlabels : bool
        If True, apply xlabel formatting according to the above options.

    Other Parameters
    ----------------
    **kwargs : See alepython.ale_plot.

    """
    predictor = (
        kwargs["model"].predict
        if kwargs.get("predictor") is None
        else kwargs["predictor"]
    )
    if zorders is None:
        zorders = list(range(len(features)))

    quantile_list = []
    ale_list = []
    mc_hull_points_list = []
    for feature in tqdm(
        features,
        desc="Calculating feature ALEs",
        disable=not kwargs.get("verbose", False),
    ):
        out = ale_plot(
            **{
                **kwargs,
                # Override certain kwargs essential to this function.
                **dict(
                    features=feature,
                    return_data=True,
                    return_mc_data=True,
                    fig=None,
                    ax=None,
                ),
            }
        )
        if len(out) == 3:
            temp_fig, _, (quantiles, ale) = out
            mc_data = None
        else:
            temp_fig, _, (quantiles, ale), mc_data = out
        # Close the unneeded temporary figure.
        plt.close(temp_fig)

        # Record the generated data for this feature.
        quantile_list.append(quantiles)
        ale_list.append(ale)

        if mc_data:
            # Add interpolated points to the list of hull points. Interpolation is
            # done to better capture the shape outlined by the lines instead of just
            # the points.
            mc_hull_points = []
            for (mc_quantiles, mc_ale) in mc_data:
                if mc_quantiles.size < 100:
                    interp_quantiles = np.linspace(
                        np.min(mc_quantiles), np.max(mc_quantiles), 100
                    )
                    interp_ale = np.interp(interp_quantiles, mc_quantiles, mc_ale)
                    mc_hull_points.append((interp_quantiles, interp_ale))
                else:
                    mc_hull_points.append((mc_quantiles, mc_ale))

            mc_hull_points_list.append(np.hstack(mc_hull_points).T)

    # Construct quantiles from the individual quantiles, minimising the amount of interpolation.
    combined_quantiles = np.vstack([quantiles[None] for quantiles in quantile_list])

    final_quantiles = np.mean(combined_quantiles, axis=0)

    mod_quantiles = np.arange(len(quantiles))

    if kwargs.get("grid_kwargs") is None:
        grid_kwargs = {}

    if kwargs.get("hull_polygon_kwargs") is None:
        hull_polygon_kwargs = {}

    if "alpha" not in hull_polygon_kwargs:
        hull_polygon_kwargs["alpha"] = 0.2

    fig = kwargs.get("fig")
    ax = kwargs.get("ax")

    if fig is None and ax is None:
        fig, ax = plt.subplots(
            figsize=(7, 3)
        )  # Make sure plot is plotted onto a new figure.
    elif fig is None:
        fig = ax.get_figure()
    if ax is None:
        ax = plt.axes()

    for feature, quantiles, ale, marker, color, zorder, mc_hull_points in zip(
        features,
        quantile_list,
        ale_list,
        markers,
        colors,
        zorders,
        (mc_hull_points_list if mc_hull_points_list else [None] * len(features)),
    ):
        if mc_hull_points is not None:
            # Interpolate the hull points to the accumulated final quantiles.
            mc_hull_points[:, 0] = np.interp(
                mc_hull_points[:, 0], final_quantiles, mod_quantiles
            )
            ax.add_patch(
                PolygonPatch(
                    alphashape(
                        mc_hull_points, alpha=kwargs.get("monte_carlo_hull_alpha")
                    ),
                    **{**hull_polygon_kwargs, **dict(facecolor=color, zorder=zorder)},
                )
            )

        # Interpolate each of the quantiles relative to the accumulated final quantiles.
        ax.plot(
            np.interp(quantiles, final_quantiles, mod_quantiles),
            ale,
            marker=marker,
            label=feature,
            c=color,
            zorder=zorder,
        )

    ax.legend(loc="best", ncol=2)

    if format_xlabels:
        ax.set_xticks(mod_quantiles[::xlabel_skip])
        ax.set_xticklabels(_sci_format(final_quantiles[::xlabel_skip], scilim=0.6))
        ax.xaxis.set_tick_params(rotation=x_rotation)
    else:
        ax.set_xticks(mod_quantiles)
        ax.set_xticklabels(final_quantiles[::xlabel_skip])

    if title is None:
        mc_string = (
            kwargs.get("monte_carlo_rep", 50) if kwargs.get("monte_carlo") else "False"
        )
        _ax_title(
            ax,
            f"First-order ALE of features '{', '.join(features)}'",
            f"Bins : {len(quantile_list[0]) - 1} - Monte-Carlo : {mc_string}",
        )
    else:
        fig.suptitle(title)
    ax.set_xlabel(xlabel, va="center_baseline")
    ax.set_ylabel(ylabel)

    if "linestyle" not in grid_kwargs:
        grid_kwargs["linestyle"] = "--"
    if "alpha" not in grid_kwargs:
        grid_kwargs["alpha"] = 0.4

    if grid_kwargs:
        ax.grid(**grid_kwargs)
