import argparse
from datetime import datetime
from multiprocessing.pool import ThreadPool
import functools
import glob
import pathlib
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
from ecl.summary import EclSum

from flownet.ert.forward_models.utils import get_last_iteration


def _load_simulations(runpath: str, ecl_base: str) -> Tuple[str, Optional[EclSum]]:
    """
    Internal helper function to load simulation results in parallel.

    Args:
        runpath: Path to where the realization is run.
        ecl_base: Path to where the realization is run.

    Returns:
        (runpath, EclSum), or (runpath, None) in case of failed simulation (inexistent .UNSMRY file)

    """
    try:
        eclsum = EclSum(str(pathlib.Path(runpath) / pathlib.Path(ecl_base)))
    except KeyboardInterrupt:
        raise
    except Exception:  # pylint: disable=broad-except
        eclsum = None
    except BaseException:  # pylint: disable=broad-except
        pass

    return runpath, eclsum


def make_dataframe_simulation_data(
    mode: str, path: str, eclbase_file: str, keys: List[str], end_date: datetime
) -> Tuple[pd.DataFrame, str, int]:
    """
    Internal helper function to generate dataframe containing
    data from ensemble of simulations from selected simulation keys

    Args:
        mode: String with mode in which flownet is run: prediction (pred) or assisted history matching (ahm)
        path: path to folder containing ensemble of simulations
        eclbase_file: name of simulation case file
        keys: list of prefix of quantities of interest to be loaded
        end_date: end date of time period for accuracy analysis

    Raises:
        ValueError: If mode is invalid (not pred or ahm).

    Returns:
        df_sim: Pandas dataframe contained data from ensemble of simulations
        iteration: current AHM iteration number
        nb_real: number of realizations

    """
    if mode == "pred":
        runpath_list = glob.glob(path)
        iteration = "latest"
    elif mode == "ahm":
        (i, runpath_list) = get_last_iteration(path)
        iteration = str(i)
    else:
        raise ValueError(
            f"{mode} is not a valid mode to run flownet with. Choose ahm or pred."
        )

    partial_load_simulations = functools.partial(
        _load_simulations, ecl_base=eclbase_file
    )

    # Load summary files of latest iteration
    realizations_dict: Dict[str, Any] = {}
    realizations_dict = dict(
        ThreadPool(processes=None).map(partial_load_simulations, runpath_list)
    )

    n_realization = 0

    # Load all simulation results for the required vector keys
    df_sim = pd.DataFrame()
    for _, eclsum in realizations_dict.items():
        if eclsum and eclsum.dates[-1] >= end_date:
            df_realization = eclsum.pandas_frame(
                column_keys=[key + "*" for key in keys]
            )
            df_realization["DATE"] = eclsum.dates
            df_realization["REALIZATION"] = n_realization

            df_sim = df_sim.append(df_realization)
            n_realization += 1

    return df_sim, iteration, n_realization


def parse_arguments():
    parser = argparse.ArgumentParser(prog=("Save ensemble predictions to CSV file."))
    parser.add_argument("mode", type=str, help="Mode: ahm or pred")
    parser.add_argument("runpath", type=str, help="Path to the ERT runpath.")
    parser.add_argument(
        "eclbase", type=str, help="Path to the simulation from runpath."
    )
    parser.add_argument(
        "end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="End date (YYYY-MM-DD) of prediction simulation.",
    )
    parser.add_argument(
        "quantity",
        type=str,
        help="List of names of quantities of interest for exporting predictions.",
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Prefix of name of CSV file containing the prediction output of the FlowNet ensemble.",
    )
    args = parser.parse_args()
    args.runpath = args.runpath.replace("%d", "*")

    return args


def save_predictions():
    """
    This function is called as a post-simulation workflow in ERT, exporting predictions
    of the FlowNet ensemble of all iterations to a CSV file in the FlowNet output folder.

    Args:
        None

    Returns:
        Nothing

    """
    args = parse_arguments()

    print("Saving predictions...", end=" ", flush=True)

    # Vector keys to export
    vector_keys = list(args.quantity.replace("[", "").replace("]", "").split(","))

    # Load ensemble of FlowNet
    (df_sim, iteration, nb_real) = make_dataframe_simulation_data(
        args.mode,
        args.runpath,
        args.eclbase,
        vector_keys,
        args.end,
    )

    # Saving dataframe to CSV file
    df_sim.to_csv(args.outfile + "_iter-" + str(iteration) + ".csv", index=False)

    print("[Done]")


if __name__ == "__main__":
    save_predictions()
