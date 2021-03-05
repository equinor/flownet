import argparse
import pathlib
import re
from typing import Optional, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from fmu import ensemble
from ecl.summary import EclSum

matplotlib.use("Agg")


def plot_ensembles(
    vector: str,
    ensembles_data: List[pd.DataFrame],
    color: Tuple[float, float, float, float],
):
    """Function to plot a list of ensembles.

    Args:
        vector: Name of the vector to plot
        ensembles_data: List of dataframes with ensemble data
        color: Color to use when plotting the ensemble
    """
    for ensemble_data in ensembles_data:
        ensemble_data = (
            remove_duplicates(ensemble_data[["DATE", "REAL", vector]])
            .pivot(index="DATE", columns="REAL", values=vector)
            .dropna()
        )

        plt.plot(
            ensemble_data.index,
            ensemble_data.values,
            color=color,
        )


def plot(
    vector: str,
    prior_data: list,
    posterior_data: list,
    reference_simulation: Optional[EclSum],
    plot_settings: dict,
):
    """Main plotting function that generate builds up a single plot build up
    from potentially multiple ensembles and other data.

    Args:
        vector: Name of the vector to plot.
        prior_data: List of prior ensemble data DataFrames.
        posterior_data: List of posterior ensemble data DataFrames.
        reference_simulation: EclSum object for the reference simulation.
        plot_settings: Settings dictionary for the plots.

    """
    plt.figure()  # (figsize=[16, 8])

    if len(prior_data):
        plot_ensembles(vector, prior_data, (0.5, 0.5, 0.5, 0.1))

    if len(posterior_data):
        plot_ensembles(vector, posterior_data, (0.0, 0.1, 0.6, 0.1))

    if reference_simulation:
        plt.plot(
            reference_simulation.dates,
            reference_simulation.numpy_vector(vector),
            color=(1.0, 0.0, 0.0, 1.0),
        )

    plt.ylim([plot_settings["ymin"], plot_settings["ymax"]])
    plt.xlabel("date")
    plt.ylabel(vector + " [" + plot_settings["units"] + "]")
    plt.savefig(re.sub(r"[^\w\-_\. ]", "_", vector), dpi=300)


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
    if not (len(args.ymin) == 1 or len(args.ymin) == len(args.vector)):
        raise ValueError(
            f"You should either supply a single minimum y-value or as many as you have vectors ({len(args.vectors)}."
        )

    if not (len(args.ymax) == 1 or len(args.ymax) == len(args.vector)):
        raise ValueError(
            f"You should either supply a single maximum y-value or as many as you have vectors ({len(args.vectors)}."
        )

    if not (len(args.units) == 1 or len(args.units) == len(args.vector)):
        raise ValueError(
            f"You should either supply a units label or as many as you have vectors ({len(args.vectors)}."
        )

    if len(args.prior) and len(args.posterior) and not args.reference_simulation:
        raise ValueError(
            "There is no prior, posterior or reference simulation to plot. Supply at least something for me to plot."
        )


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
        help="One or more vectors to plot separated by spaces. Example: WOPR:WELL1 FOPR",
    )
    parser.add_argument(
        "-prior",
        type=str,
        nargs="+",
        help="One or more paths to prior ensembles separated by a space."
        "The path should include a '%d' which indicates the realization number.",
    )
    parser.add_argument(
        "-posterior",
        type=str,
        nargs="+",
        help="One or more paths to posterior ensembles separated by a space."
        "The path should include a '%d' which indicates the realization number.",
    )
    parser.add_argument(
        "-reference_simulation",
        "-r",
        type=pathlib.Path,
        help="Path to the reference simulation case.",
    )
    parser.add_argument(
        "-ymin",
        type=float,
        default=0,
        nargs="+",
        help="One or #vectors minimum y values.",
    )
    parser.add_argument(
        "-ymax",
        type=float,
        default=1000,
        nargs="+",
        help="One or #vectors maximum y values.",
    )
    parser.add_argument(
        "-units",
        type=str,
        default="Cows/Lightyear",
        nargs="+",
        help="One or #vectors unit labels.",
    )
    args = parser.parse_args()

    check_args(args)

    prior_data: list = []
    for prior in args.prior:
        prior_data.append(
            ensemble.ScratchEnsemble(
                "flownet_ensemble",
                paths=prior.replace("%d", "*"),
            ).get_smry(column_keys=args.vectors)
        )

    posterior_data: list = []
    for posterior in args.posterior:
        posterior_data.append(
            ensemble.ScratchEnsemble(
                "flownet_ensemble", paths=posterior.replace("%d", "*")
            ).get_smry(column_keys=args.vectors)
        )

    reference_eclsum = EclSum(str(args.reference_simulation.with_suffix(".UNSMRY")))

    for i, vector in enumerate(args.vectors):

        plot_settings = {
            "ymin": args.ymin[0] if len(args.ymin) == 1 else args.ymin[i],
            "ymax": args.ymax[0] if len(args.ymax) == 1 else args.ymax[i],
            "units": args.units[0] if len(args.units) == 1 else args.units[i],
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
