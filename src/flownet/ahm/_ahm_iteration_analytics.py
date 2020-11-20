import argparse
import glob
import itertools
import os
import pathlib
import re
from typing import Any, Dict, Optional, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ecl.summary import EclSum

from ..data import FlowData


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
) -> np.array:
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
) -> np.array:
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
    data_opm_reference: np.array, data_ensembles_flownet: List[np.array]
) -> Tuple[np.array, List[np.array]]:
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
        lowNet simulations
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
    data_reference: pd.DataFrame, data_test: np.array, metric: str
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


def _load_simulations(runpath: str, ecl_base: str) -> Optional[EclSum]:
    """
    Internal helper function to simulation results in parallel.
    Args:
        runpath: Path to where the realization is run.
        ecl_base: Path to where the realization is run.
    Returns:
        EclSum, or None in case of failed simulation (inexistent .UNSMRY file)
    """
    if os.path.exists(pathlib.Path(runpath) / pathlib.Path(ecl_base + ".UNSMRY")):
        return EclSum(str(pathlib.Path(runpath) / pathlib.Path(ecl_base)))
    return None


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
    obs: np.array,
    list_ensembles: List[np.array],
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
    path: str, eclbase_file: str, keys: List[str], end_date: np.datetime64
) -> Tuple[pd.DataFrame, int, int]:
    """
    Internal helper function to generate dataframe containing
    data from ensemble of simulations from selected simulation keys

    Args:
        path: path to folder containing ensemble of simulations
        eclbase_file: name of simulation case file
        keys: list of prefix of quantities of interest to be loaded
        end_date: end date of time period for accuracy analysis

    Returns:
        df_sim: Pandas dataframe contained data from ensemble of simulations
        iteration: current AHM iteration number
        nb_real: number of realizations
    """

    iteration = int(re.findall(r"[0-9]+", sorted(glob.glob(path))[-1])[-1])
    runpath_list = glob.glob(path[::-1].replace("*", str(iteration), 1)[::-1])

    realizations_dict = {}
    # Load summary files of latest iteration
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    for runpath, eclbase in list(zip(runpath_list, itertools.repeat(eclbase_file))):
        realizations_dict[runpath] = _load_simulations(runpath, eclbase)

    # Prepare dataframe
    # pylint: disable-msg=too-many-locals
    df_sim = pd.DataFrame()
    nb_real = 0
    for runpath in realizations_dict:
        df_tmp = pd.DataFrame()
        if realizations_dict[runpath] and (
            np.datetime64(realizations_dict[runpath].dates[-1]) >= end_date
        ):
            dates = realizations_dict[runpath].dates
            if nb_real == 0:
                df_sim["DATE"] = pd.Series(dates)
                df_sim["REAL_ID"] = pd.Series(nb_real * np.ones(len(dates)), dtype=int)
            df_tmp["DATE"] = pd.Series(dates)
            df_tmp["REAL_ID"] = pd.Series(nb_real * np.ones(len(dates)), dtype=int)

            if nb_real == 0:
                for counter, k in enumerate(keys):
                    if counter == 0:
                        key_list_data = [
                            x for x in realizations_dict[runpath] if x.startswith(k)
                        ]
                    else:
                        key_list_data.extend(
                            [x for x in realizations_dict[runpath] if x.startswith(k)]
                        )

            for key in key_list_data:
                if nb_real == 0:
                    df_sim[key] = pd.Series(
                        realizations_dict[runpath].numpy_vector(key)
                    )
                df_tmp[key] = pd.Series(realizations_dict[runpath].numpy_vector(key))

            if nb_real > 0:
                df_sim = df_sim.append(df_tmp, ignore_index=True)

            nb_real = nb_real + 1

    return df_sim, iteration, nb_real


def save_plots_metrics(df_metrics: pd.DataFrame, metrics: List[str], str_key: str):
    """
    Internal helper function to generate and save plots of evolution of
    accuracy metrics over iterations

    Args:
        df_metrics: Pandas dataframe containing values of accuracy metrics over iterations
        metrics: list containing names of computed accuracy metrics
        str_key: name of quantity of interest for accuracy metric calculation

    Returns:
        Nothing
    """

    tmp_df_plot = df_metrics[df_metrics["quantity"] == str_key]
    min_it = np.amin(tmp_df_plot["iteration"].values)
    max_it = np.amax(tmp_df_plot["iteration"].values)
    for acc_metric in metrics:
        plt.figure(figsize=(6, 4))
        plt.xlabel("Iterations")
        plt.ylabel(acc_metric.upper() + " history")
        plt.plot(tmp_df_plot["iteration"], tmp_df_plot[acc_metric])
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        if min_it != max_it:
            plt.xlim((min_it, max_it))
        plt.savefig("metric_" + acc_metric + "_" + str_key + ".png")


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

    parser = argparse.ArgumentParser(prog=("Save iteration analytics to a file."))
    parser.add_argument(
        "reference_simulation", type=str, help="Path to the reference simulation case"
    )
    parser.add_argument(
        "perforation_strategy", type=str, help="Perforation handling strategy"
    )
    parser.add_argument("runpath", type=str, help="Path to the ERT runpath.")
    parser.add_argument(
        "eclbase", type=str, help="Path to the simulation from runpath."
    )
    parser.add_argument(
        "start", type=str, help="Start date (YYYY-MM-DD) for accuracy analysis."
    )
    parser.add_argument(
        "end", type=str, help="End date (YYYY-MM-DD) for accuracy analysis."
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

    print("Saving iteration analytics...", end=" ")

    # Fix list inputs
    metrics = list(args.metrics.replace("[", "").replace("]", "").split(","))

    # Load ensemble of FlowNet
    (df_sim, iteration, nb_real) = make_dataframe_simulation_data(
        args.runpath,
        args.eclbase,
        list(args.quantity.replace("[", "").replace("]", "").split(",")),
        np.datetime64(args.end),
    )

    # Load reference simulation (OPM-Flow/Eclipse)
    field_data = FlowData(
        args.reference_simulation,
        perforation_handling_strategy=args.perforation_strategy,
    )
    df_obs: pd.DataFrame = field_data.production
    df_obs["DATE"] = df_obs["date"]

    df_obs["DATE"] = pd.to_datetime(df_obs["DATE"])
    df_sim["DATE"] = pd.to_datetime(df_sim["DATE"], format="%Y-%m-%dT%H:%M:%S")

    # Filter dataframe base on measurement dates
    df_obs = df_obs[df_obs["DATE"].isin(df_sim["DATE"])]
    df_sim = df_sim[df_sim["DATE"].isin(df_obs["DATE"])]

    # Initiate dataframe with metrics
    df_metrics = load_csv_file(args.outfile, ["quantity", "iteration"] + metrics)

    for str_key in list(args.quantity.replace("[", "").replace("]", "").split(",")):
        # Prepare data from reference simulation
        tmp_data = filter_dataframe(
            df_obs,
            "DATE",
            np.datetime64(args.start),
            np.datetime64(args.end),
        )

        truth_data = pd.DataFrame()
        truth_data["DATE"], unique_idx = np.unique(
            tmp_data["DATE"].values, return_index=True
        )
        truth_data["REAL_ID"] = pd.Series(np.zeros((len(unique_idx))), dtype=int)
        for well in np.unique(tmp_data["WELL_NAME"]):
            truth_data[str_key[:5] + well] = np.zeros((len(unique_idx), 1))
            truth_data.iloc[
                [
                    idx
                    for idx, val in enumerate(truth_data["DATE"].values)
                    if val in tmp_data[tmp_data["WELL_NAME"] == well]["DATE"].values
                ],
                truth_data.columns.get_loc(str_key[:5] + well),
            ] = tmp_data[tmp_data["WELL_NAME"] == well][str_key[:4]].values

        truth_data = truth_data.fillna(0)
        truth_data = truth_data[list(set(truth_data.columns) & set(df_sim.columns))]

        obs_opm = prepare_opm_reference_data(truth_data, str_key, nb_real)

        # Prepare data from ensemble of FlowNet
        ens_flownet = []
        ens_flownet.append(
            prepare_flownet_data(
                filter_dataframe(
                    df_sim[truth_data.columns],
                    "DATE",
                    np.datetime64(args.start),
                    np.datetime64(args.end),
                ),
                str_key,
                nb_real,
            )
        )

        # Normalizing data
        obs_opm, ens_flownet = normalize_data(obs_opm, ens_flownet)

        # Appending dataframe with accuracy metrics of current iteration
        df_metrics = df_metrics.append(
            compute_metric_ensemble(obs_opm, ens_flownet, metrics, str_key, iteration),
            ignore_index=True,
        )

        # Plotting accuracy metrics over iterations
        save_plots_metrics(df_metrics, metrics, str_key)

    # Saving accuracy metrics to CSV file
    df_metrics.to_csv(args.outfile + ".csv", index=False)

    print("[Done]")
