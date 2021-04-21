import argparse
import pathlib
import re
from datetime import datetime
from typing import Optional, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from fmu import ensemble
from ecl.summary import EclSum

from .observations import _read_ert_obs

matplotlib.use("Agg")


def plot_ensembles(
    ensemble_type: str,
    vector: str,
    ensembles_data: List[pd.DataFrame],
    plot_settings: dict,
):
    """Function to plot a list of ensembles.

    Args:
        ensemble_type: prior or posterior
        vector: Name of the vector to plot
        ensembles_data: List of dataframes with ensemble data
        plot_settings: Settings dictionary for the plots.

    Returns:
        Nothing

    Raises:
        Value error if incorrect plot type.

    """
    if not ensemble_type in ("prior", "posterior"):
        raise ValueError("Plot type should be either prior or posterior.")

    for i, ensemble_data in enumerate(ensembles_data):

        ensemble_data = (
            remove_duplicates(ensemble_data[["DATE", "REAL", vector]])
            .pivot(index="DATE", columns="REAL", values=vector)
            .dropna()
        )

        color = (
            plot_settings[f"{ensemble_type}_colors"][0]
            if len(plot_settings[f"{ensemble_type}_colors"]) == 1
            else plot_settings[f"{ensemble_type}_colors"][i]
        )
        alpha = (
            plot_settings[f"{ensemble_type}_alphas"][0]
            if len(plot_settings[f"{ensemble_type}_alphas"]) == 1
            else plot_settings[f"{ensemble_type}_alphas"][i]
        )

        plt.plot(
            ensemble_data.index,
            ensemble_data.values,
            color=color,
            alpha=alpha,
            linestyle="solid",
        )


def plot(
    vector: str,
    prior_data: list,
    posterior_data: list,
    reference_simulation: Optional[EclSum],
    plot_settings: dict,
):
    """Main plotting function that generates a single plot
    from potentially multiple ensembles and other data.

    Args:
        vector: Name of the vector to plot.
        prior_data: List of prior ensemble data DataFrames.
        posterior_data: List of posterior ensemble data DataFrames.
        reference_simulation: EclSum object for the reference simulation.
        plot_settings: Settings dictionary for the plots.

    """
    plt.figure()  # (figsize=[16, 8])

    if prior_data:
        plot_ensembles("prior", vector, prior_data, plot_settings)

    if posterior_data:
        plot_ensembles("posterior", vector, posterior_data, plot_settings)

    if reference_simulation:
        plt.plot(
            reference_simulation.dates,
            reference_simulation.numpy_vector(vector),
            color=plot_settings["reference_simulation_color"],
            alpha=1,
        )

    if plot_settings["vertical_lines"] is not None:
        for vertical_line_date in plot_settings["vertical_lines"]:
            plt.axvline(x=vertical_line_date, color="k", linestyle="--")

    if plot_settings["errors"] is not None:
        if vector in plot_settings["errors"]:
            plt.errorbar(
                plot_settings["errors"][vector][0],
                plot_settings["errors"][vector][1],
                yerr=plot_settings["errors"][vector][2],
                fmt="o",
                color="k",
                ecolor="k",
                capsize=5,
                elinewidth=2,
            )

    plt.ylim([plot_settings["ymin"], plot_settings["ymax"]])
    plt.xlabel("date")
    plt.ylabel(vector + " [" + plot_settings["units"] + "]")
    plt.savefig(re.sub(r"[^\w\-_\. ]", "_", vector), dpi=300)
    plt.close()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates for the combination or DATE and REAL.

    Args:
        df: Input pandas DataFrame with columns: [DATE, REAL, VECTOR1, VECTOR2, ..., VECTOR_N]

    Returns:
        A cleaned dataframe

    """
    return df[~df[["DATE", "REAL"]].apply(frozenset, axis=1).duplicated()]


def check_args(args):
    """Helper function to verify input arguments.

    Returns:
        Nothing

    Raises:
        ValueError in case the input arguments are inconsistent.

    """
    if not (len(args.ymin) == 1 or len(args.ymin) == len(args.vectors)):
        raise ValueError(
            f"You should either supply a single minimum y-value or as many as you have vectors ({len(args.vectors)}."
        )

    if not (len(args.ymax) == 1 or len(args.ymax) == len(args.vectors)):
        raise ValueError(
            f"You should either supply a single maximum y-value or as many as you have vectors ({len(args.vectors)}."
        )

    if not (len(args.units) == 1 or len(args.units) == len(args.vectors)):
        raise ValueError(
            f"You should either supply a single units label or as many as you have vectors ({len(args.vectors)}."
        )

    if (
        args.prior is None
        and args.posterior is None
        and args.reference_simulation is None
    ):
        raise ValueError(
            "There is no prior, no posterior and no reference simulation to plot. Supply at least at one "
            "of the three to plot."
        )

    if not (len(args.prior_colors) == 1 or len(args.prior_colors) == len(args.prior)):
        raise ValueError(
            "You should either supply a single prior color or as "
            f"many as you have prior distributions ({len(args.prior)}."
        )

    if not (
        len(args.posterior_colors) == 1
        or len(args.posterior_colors) == len(args.posterior)
    ):
        raise ValueError(
            "You should either supply a single posterior color or as "
            f"many as you have posterior distributions ({len(args.posterior)}."
        )


def build_ensemble_df_list(
    ensemble_paths: Optional[List[str]], vectors: List[str]
) -> List[pd.DataFrame]:
    """Helper function to read and prepare ensemble data.

    Args:
        ensemble_paths: The ensemble paths to retrieve data from
        vectors: List of vector to extract

    Returns:
        List of ensemble dataframe with required data to create plots.

    """
    data: list = []

    if ensemble_paths is not None:
        for prior in ensemble_paths:

            df_data = ensemble.ScratchEnsemble(
                "flownet_ensemble",
                paths=prior.replace("%d", "*"),
            ).get_smry(column_keys=vectors)

            df_data_sorted = df_data.sort_values("DATE")
            df_realizations = df_data_sorted[
                df_data_sorted["DATE"] == df_data_sorted.values[-1][0]
            ]["REAL"]

            data.append(df_data.merge(df_realizations, how="inner"))

    return data


def main():
    """Main function for the plotting of simulations results from FlowNet.

    Return:
        Nothing
    """

    parser = argparse.ArgumentParser(
        prog=("Simple tool to plot FlowNet ensembles simulation results.")
    )
    parser.add_argument(
        "vectors",
        type=str,
        nargs="+",
        help="One or more vectors to plot separated by spaces. "
        "Example: WOPR:WELL1 FOPR",
    )
    parser.add_argument(
        "-prior",
        type=str,
        nargs="+",
        default=None,
        help="One or more paths to prior ensembles separated by a space. "
        "The path should include a '%d' which indicates the realization number. "
        "Example: runpath/realization-%d/iter-0/",
    )
    parser.add_argument(
        "-posterior",
        type=str,
        nargs="+",
        default=None,
        help="One or more paths to posterior ensembles separated by a space. "
        "The path should include a '%d' which indicates the realization number. "
        "Example: runpath/realization-%d/iter-4/",
    )
    parser.add_argument(
        "-reference_simulation",
        "-r",
        type=pathlib.Path,
        default=None,
        help="Path to the reference simulation case. "
        "Example: path/to/SIMULATION.DATA",
    )
    parser.add_argument(
        "-ymin",
        type=float,
        default=[0],
        nargs="+",
        help="Lower cut-off of the y-axis. Can be one number or multiple values, "
        "depending on the number of vectors you are plotting. In the latter case, "
        "the number of y-min values should be equal to the number of vectors.",
    )
    parser.add_argument(
        "-ymax",
        type=float,
        default=[1000],
        nargs="+",
        help="Upper cut-off of the y-axis. Can be one number or multiple values, "
        "depending on the number of vectors you are plotting. In the latter case, "
        "the number of y-max values should be equal to the number of vectors.",
    )
    parser.add_argument(
        "-units",
        type=str,
        default=[""],
        nargs="+",
        help="Unit label for the y-axis. Can be one number or multiple units, "
        "depending on the number of vectors you are plotting. In the latter case, "
        "the number of units should be equal to the number of vectors.",
    )
    parser.add_argument(
        "-prior_alphas",
        type=float,
        default=[0.1],
        nargs="+",
        help="Transparency of prior, value between 0 (transparent) and 1 (opaque). "
        "Can be one number or multiple values, depending on the number of priors you "
        "are plotting. In the latter case, the number of alpha values should be equal "
        "to the number of priors.",
    )
    parser.add_argument(
        "-posterior_alphas",
        type=float,
        default=[0.1],
        nargs="+",
        help="Transparency of posterior, value between 0 (transparent) and 1(opaque). "
        "Can be one number or multiple values, depending on the number of posteriors "
        "you are plotting. In the latter case, the number of alpha values should be equal "
        "to the number of posteriors.",
    )
    parser.add_argument(
        "-prior_colors",
        type=str,
        default=["gray"],
        nargs="+",
        help="Color of prior lines. Can be one number or multiple colors, depending on "
        "the number of priors you are plotting. In the latter case, the number of colors "
        "should be equal to the number of priors.",
    )
    parser.add_argument(
        "-posterior_colors",
        type=str,
        default=["blue"],
        nargs="+",
        help="Color of posterior. Can be one number or colors, depending on the number "
        "of posteriors you are plotting. In the latter case, the number of colors should "
        "be equal to the number of posteriors.",
    )
    parser.add_argument(
        "-reference_simulation_color",
        type=str,
        default="red",
        help="The reference simulation color. Examples: 'red', 'blue', 'green'.",
    )
    parser.add_argument(
        "-vertical_lines",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=None,
        nargs="+",
        help="One or more dates (YYYY-MM-DD) to add vertical lines in the plot.",
    )
    parser.add_argument(
        "-ertobs",
        type=pathlib.Path,
        default=None,
        help="Path to an ERT observation file.",
    )
    args = parser.parse_args()

    check_args(args)

    prior_data = build_ensemble_df_list(args.prior, args.vectors)
    posterior_data = build_ensemble_df_list(args.posterior, args.vectors)

    if args.ertobs is not None:
        ertobs = _read_ert_obs(args.ertobs)
    else:
        ertobs = None

    if args.reference_simulation is not None:
        reference_eclsum = EclSum(str(args.reference_simulation.with_suffix(".UNSMRY")))
    else:
        reference_eclsum = None

    for i, vector in enumerate(args.vectors):

        plot_settings = {
            "ymin": args.ymin[0] if len(args.ymin) == 1 else args.ymin[i],
            "ymax": args.ymax[0] if len(args.ymax) == 1 else args.ymax[i],
            "units": args.units[0] if len(args.units) == 1 else args.units[i],
            "prior_alphas": args.prior_alphas,
            "posterior_alphas": args.posterior_alphas,
            "prior_colors": args.prior_colors,
            "posterior_colors": args.posterior_colors,
            "reference_simulation_color": args.reference_simulation_color,
            "vertical_lines": args.vertical_lines,
            "errors": ertobs,
        }

        print(f"Plotting {vector}...", end=" ", flush=True)

        plot(
            vector,
            prior_data,
            posterior_data,
            reference_eclsum,
            plot_settings,
        )

        print("[Done]", flush=True)


if __name__ == "__main__":
    main()
