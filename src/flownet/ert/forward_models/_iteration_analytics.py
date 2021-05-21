import argparse
from datetime import datetime
from multiprocessing.pool import ThreadPool
import functools
import glob
import os
import pathlib
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ecl.summary import EclSum

from flownet.data import FlowData
from flownet.ert.forward_models.utils import get_last_iteration


def filter_dataframe(
    df_in: pd.DataFrame, key_filter: str, min_value: Any, max_value: Any
) -> pd.DataFrame:
    """
    This function filters the rows of a Pandas dataframe
    based on a range of values of a specified column

    Args:
        df_in: is the Pandas dataframe to be filtered
        key_filter: is the key of the specified column
        for which the dataframe is to be filtered
        min_value: is the minimum value of the range for filtering
        max_value: is the maximum value of the range for filtering

    Returns:
        A Pandas dataframe containing only values within the specified range
        for the provided column

    """
    return df_in[(df_in[key_filter] >= min_value) & (df_in[key_filter] < max_value)]


def prepare_opm_reference_data(
    df_opm: pd.DataFrame, str_key: str, n_real: int
) -> np.ndarray:
    """
    This function extracts data from selected columns of the Pandas dataframe
    containing data from reference simulation, rearranges it into a stacked
    column vector preserving the original order and repeats it n_real times to form
    a matrix for comparison with data from ensemble of n_real FlowNet simulations

    Args:
        df_opm: is the Pandas dataframe containing data from reference simulation
        str_key: is the string to select columns; column names starting with str_key
        n_real: is the size of ensemble of FlowNet simulations

    Returns:
        A numpy 2D array [length_data * nb_selected_columns, n_real] containing data
        from selected columns (i.e., quantity of interest for accuracy metric) of
        reference simulation stacked in a column-vector and replicated into n_real columns

    """
    keys = df_opm.keys()
    keys = sorted(keys[df_opm.keys().str.contains(str_key)])
    data = np.transpose(np.tile(df_opm[keys].values.flatten(), (n_real, 1)))

    return data


def prepare_flownet_data(
    df_flownet: pd.DataFrame, str_key: str, n_real: int
) -> np.ndarray:
    """
    This function extracts data from selected columns of the Pandas dataframe
    containing data from an ensemble of FlowNet simulations, rearranges it into
    a matrix of stacked column-vectors preserving the original order, i.e. one column
    per realization of the ensemble

    Args:
        df_flownet: is the Pandas dataframe containing data from ensemble of FlowNet simulations
        str_key: is the string to select columns; column names starting with str_key
        n_real: is the size of ensemble of FlowNet simulations

    Returns:
        A numpy 2D array [length_data * nb_selected_columns, n_real] containing data
        from selected columns (i.e., quantity of interest for accuracy metric) for
        an ensemble of FlowNet simulations in a column-vector. Each column correspond
        to one realization of the ensemble

    """
    keys = df_flownet.keys()
    keys = sorted(keys[df_flownet.keys().str.contains(str_key)])
    data = df_flownet[keys].values.flatten()
    data = np.reshape(data, (data.shape[0] // n_real, n_real), order="F")

    return data


def normalize_data(
    data_opm_reference: np.ndarray, data_ensembles_flownet: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    This function normalizes data from a 2D numpy array containing data from the
    reference simulation and multiple ensembles of FlowNet simulations,
    using the MinMaxScaler from sklearn.preprocessing module

    Args:
        data_opm_reference: is the 2D numpy array containing data from reference
        simulation replicated into as many columns as the size of ensemble of
        FlowNet realizations
        data_ensembles_flownet: is a list of 2D numpy arrays containing data from
        ensembles of FlowNet simulations; each list entry correspond to the ensemble of
        a given iteration of ES-MDA

    Returns:
        norm_data_opm_reference: is a normalized 2D numpy array for the reference simulation data
        norm_data_ensembles_flownet: a list of normalized 2D numpy arrays for the ensembles of
        FlowNet simulations

    """
    for k, data_ens in enumerate(data_ensembles_flownet):
        if k == 0:
            tmp = data_ens
        else:
            tmp = np.append(tmp, data_ens, axis=1)

    matrix_data = np.append(data_opm_reference, tmp, axis=1)

    if np.isclose(data_opm_reference.max(), data_opm_reference.min()):
        if np.isclose(data_opm_reference.max(), 0.0):
            scale = np.tile(1.0, matrix_data.shape[0])
        else:
            scale = 1 / (np.tile(data_opm_reference.max(), matrix_data.shape[0]))
    else:
        scale = 1 / (
            np.tile(data_opm_reference.max(), matrix_data.shape[0])
            - np.tile(data_opm_reference.min(), matrix_data.shape[0])
        )

    norm_matrix_data = (
        matrix_data * scale[:, None]
        - (np.tile(data_opm_reference.min(), matrix_data.shape[0]) * scale)[:, None]
    )

    n_data = int(norm_matrix_data.shape[1] / (len(data_ensembles_flownet) + 1))
    norm_data_opm_reference = norm_matrix_data[:, :n_data]
    norm_data_ensembles_flownet = []
    for k in range(len(data_ensembles_flownet)):
        norm_data_ensembles_flownet.append(
            norm_matrix_data[:, (k + 1) * n_data : (k + 2) * n_data]
        )

    return norm_data_opm_reference, norm_data_ensembles_flownet


def accuracy_metric(
    data_reference: pd.DataFrame, data_test: np.ndarray, metric: str
) -> float:
    """
    This function computes a score value based on the specified accuracy metric
    by calculating discrepancy between columns of two 2D numpy arrays:
    (1) reference simulation data and (2) ensemble of FlowNet simulations

    Args:
        data_reference: is the 2D numpy array containing normalized data from reference
        simulation replicated into as many columns as the size of ensemble of
        FlowNet realizations
        data_test: is the 2D numpy array containing normalized data from an ensemble of
        FlowNet simulations
        metric: is a string with the name of the metric to be calculated

    Returns:
        A score value reflecting the accuracy of FlowNet simulations in terms of matching
        data from reference simulation

    """
    if metric == "MSE":
        score = mean_squared_error(data_reference, data_test)
    elif metric == "RMSE":
        score = mean_squared_error(data_reference, data_test, squared=False)
    elif metric == "NRMSE":
        score = mean_squared_error(data_reference, data_test, squared=False) / (
            np.amax(data_reference) - np.amin(data_reference)
        )
    elif metric == "MAE":
        score = mean_absolute_error(data_reference, data_test)
    elif metric == "NMAE":
        score = mean_absolute_error(data_reference, data_test) / (
            np.amax(data_reference) - np.amin(data_reference)
        )
    elif metric == "R2":
        score = r2_score(data_reference, data_test, multioutput="variance_weighted")
    else:
        raise ValueError(f"The metric {metric} is not supported.")

    return score


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


def load_csv_file(csv_file: str, csv_columns: List[str]) -> pd.DataFrame:
    """
    Internal helper function to generate dataframe containing
    selected observations and respective dates

    Args:
        csv_file: name of CSV file
        csv_columns: list of names of columns of CSV file

    Returns:
        Pandas dataframe containing data from existing CSV file or
        empty dataframe with requested columns if CSV file does not exist

    """
    if os.path.exists(csv_file + ".csv"):
        df_csv = pd.read_csv(csv_file + ".csv")
    else:
        df_csv = pd.DataFrame(columns=csv_columns)

    return df_csv


def compute_metric_ensemble(
    obs: np.ndarray,
    list_ensembles: List[np.ndarray],
    metrics: List[str],
    str_key: str,
    iteration: int,
) -> Dict:
    """
    Internal helper function to generate dataframe containing
    selected observations and respective dates

    Args:
        obs: numpy array containing normalized data from reference simulation
        replicated over as many columns as size of ensemble of FlowNet simulations
        list_ensembles: list of ensembles of FlowNet simulations; can be used if
        ensembles of multiple AHM iterations are loaded
        metrics: list of accuracy metrics to be computed
        str_key: name of quantity of interest for accuracy metric calculation
        iteration: current AHM iteration number

    Returns:
        Dictionary containing values of calculated accuracy metrics for selected
        quantity of interest for current iteration of AHM

    """
    dict_metric = {"quantity": str_key, "iteration": iteration}
    for acc_metric in metrics:
        for obs_sim in list_ensembles:
            dict_metric[acc_metric] = accuracy_metric(obs, obs_sim, acc_metric)

    return dict_metric


def make_dataframe_simulation_data(
    mode: str, path: str, eclbase_file: str, keys: List[str], end_date: datetime
) -> Tuple[pd.DataFrame, str, int]:
    """
    Internal helper function to generate dataframe containing
    data from ensemble of simulations from selected simulation keys

    Args:
        mode: String with mode in which flownet is run: prediction (pred) or assisted hisotory matching (ahm)
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

            df_sim = df_sim.append(df_realization)
            n_realization += 1

    return df_sim, iteration, n_realization


def parse_arguments():
    parser = argparse.ArgumentParser(prog=("Save iteration analytics to a file."))
    parser.add_argument("mode", type=str, help="Mode: ahm or pred")
    parser.add_argument(
        "reference_simulation", type=str, help="Path to the reference simulation case"
    )
    parser.add_argument("runpath", type=str, help="Path to the ERT runpath.")
    parser.add_argument(
        "eclbase", type=str, help="Path to the simulation from runpath."
    )
    parser.add_argument(
        "start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Start date (YYYY-MM-DD) for accuracy analysis.",
    )
    parser.add_argument(
        "end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="End date (YYYY-MM-DD) for accuracy analysis.",
    )
    parser.add_argument(
        "quantity",
        type=str,
        help="List of names of quantities of interest for accuracy analysis.",
    )
    parser.add_argument("metrics", type=str, help="List of names of accuracy metrics.")
    parser.add_argument(
        "outfile",
        type=str,
        help="Name of output file containing metrics over iterations.",
    )
    args = parser.parse_args()
    args.runpath = args.runpath.replace("%d", "*")

    return args


def save_iteration_analytics():
    """
    This function is called as a post-simulation workflow in ERT, saving all
    accuracy metrics of all iterations to a file and plotting evolution of accuracy
    metrics over iterations. The resulting accuracy metric values are stored in
    a CSV file in the FlowNet output folder, along with the figures

    Args:
        None

    Returns:
        Nothing

    """
    args = parse_arguments()

    print("Saving iteration analytics...", end=" ", flush=True)

    # Fix list inputs
    metrics = list(args.metrics.replace("[", "").replace("]", "").split(","))

    # Vector keys to analyze
    vector_keys = list(args.quantity.replace("[", "").replace("]", "").split(","))

    # Load ensemble of FlowNet
    (df_sim, iteration, nb_real) = make_dataframe_simulation_data(
        args.mode,
        args.runpath,
        args.eclbase,
        vector_keys,
        args.end,
    )

    # Load reference simulation (OPM-Flow/Eclipse)
    field_data = FlowData(args.reference_simulation)
    df_obs: pd.DataFrame = field_data.production
    df_obs["DATE"] = df_obs["date"]

    df_obs["DATE"] = pd.to_datetime(df_obs["DATE"])
    df_sim["DATE"] = pd.to_datetime(df_sim["DATE"], format="%Y-%m-%dT%H:%M:%S")

    # Filter dataframe base on measurement dates
    df_obs = df_obs[df_obs["DATE"].isin(df_sim["DATE"])]
    df_sim = df_sim[df_sim["DATE"].isin(df_obs["DATE"])]

    # Initiate dataframe with metrics
    df_metrics = load_csv_file(args.outfile, ["quantity", "iteration"] + metrics)

    # Prepare data from reference simulation
    df_obs_filtered = filter_dataframe(
        df_obs,
        "DATE",
        args.start,
        args.end,
    )

    for key in vector_keys:

        truth_data = (
            df_obs_filtered.pivot(
                index="DATE", columns="WELL_NAME", values=key.split(":")[0]
            )
            .add_prefix(key.split(":")[0] + ":")
            .fillna(0)
            .reset_index()
        )

        obs_opm = prepare_opm_reference_data(truth_data, key, nb_real)
        truth_data = truth_data.loc[:, truth_data.columns.str.startswith(key)]

        # Prepare data from ensemble of FlowNet
        ens_flownet = []
        ens_flownet.append(
            prepare_flownet_data(
                filter_dataframe(
                    df_sim[list(truth_data.columns) + ["DATE"]],
                    "DATE",
                    args.start,
                    args.end,
                ),
                key,
                nb_real,
            )
        )

        # Normalizing data
        obs_opm, ens_flownet = normalize_data(obs_opm, ens_flownet)

        # Appending dataframe with accuracy metrics of current iteration
        df_metrics = df_metrics.append(
            compute_metric_ensemble(obs_opm, ens_flownet, metrics, key, iteration),
            ignore_index=True,
        )

    # Saving accuracy metrics to CSV file
    df_metrics.to_csv(args.outfile + ".csv", index=False)

    print("[Done]")


if __name__ == "__main__":
    save_iteration_analytics()
