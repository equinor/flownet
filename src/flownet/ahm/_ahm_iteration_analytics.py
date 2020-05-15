import argparse
import glob
import itertools
import math
import os
import pathlib
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml

from ecl.summary import EclSum


def filter_dataframe(df_in, key_filter, min_value, max_value):
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

    return df_in[
        (df_in[key_filter].values >= min_value) & (df_in[key_filter].values < max_value)
    ]


def prepare_opm_reference_data(df, str_key, n):
    """
    This function extracts data from selected columns of the Pandas dataframe
    containing data from reference simulation, rearranges it into a stacked
    column vector preserving the original order and repeats it n times to form
    a matrix for comparison with data from ensemble of n FlowNet simulations

    Args:
        df: is the Pandas dataframe containing data from reference simulation
        str_key: is the string to select columns; column names starting with str_key
        n: is the size of ensemble of FlowNet simulations

    Returns:
        A numpy 2D array [length_data * nb_selected_columns, n] containing data
        from selected columns (i.e., quantity of interest for accuracy metric) of
        reference simulation stacked in a column-vector and replicated into n columns
    """

    keys = df.keys()
    keys = keys[df.keys().str.contains(str_key)]
    data = df[keys].values
    data = data.flatten()
    #data = np.reshape(
    #    df[keys].values,
    #    (df[keys].values.shape[0] * df[keys].values.shape[1], 1),
    #    order="F",
    #)
    data = np.transpose(np.tile(data, (n, 1)))

    return data


def prepare_flownet_data(df, str_key, n):
    """
    This function extracts data from selected columns of the Pandas dataframe
    containing data from an ensemble of FlowNet simulations, rearranges it into
    a matrix of stacked column-vectors preserving the original order, i.e. one column
    per realization of the ensemble

    Args:
        df: is the Pandas dataframe containing data from ensemble of FlowNet simulations
        str_key: is the string to select columns; column names starting with str_key
        n: is the size of ensemble of FlowNet simulations

    Returns:
        A numpy 2D array [length_data * nb_selected_columns, n] containing data
        from selected columns (i.e., quantity of interest for accuracy metric) for
        an ensemble of FlowNet simulations in a column-vector. Each column correspond
        to one realization of the ensemble
    """

    keys = df.keys()
    keys = keys[df.keys().str.contains(str_key)]
    data = df[keys].values
    data = data.flatten()
    data = np.reshape(data, (int(data.shape[0] / n), n), order="F")

    return data


def normalize_data(data_opm_reference, data_ensembles_flownet):
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
    scale = 1 / (
        np.tile(data_opm_reference.max(), matrix_data.shape[0])
        - np.tile(data_opm_reference.min(), matrix_data.shape[0])
    )
    norm_matrix_data = (
        matrix_data * scale[:, None]
        - (np.tile(data_opm_reference.min(), matrix_data.shape[0]) * scale)[:, None]
    )

    n = int(norm_matrix_data.shape[1] / (len(data_ensembles_flownet) + 1))
    norm_data_opm_reference = norm_matrix_data[:, :n]
    norm_data_ensembles_flownet = []
    for k in range(len(data_ensembles_flownet)):
        norm_data_ensembles_flownet.append(
            norm_matrix_data[:, (k + 1) * n : (k + 2) * n]
        )

    return norm_data_opm_reference, norm_data_ensembles_flownet


def accuracy_metric(data_reference, data_test, metric):
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

    Returns:
        A score value reflecting the accuracy of FlowNet simulations in terms of matching
        data from reference simulation
    """

    # if metric == "MSE" or metric == "RMSE" or metric == "NRMSE":
    if metric in ("MSE", "RMSE", "NRMSE"):
        score = mean_squared_error(data_reference, data_test)
        if metric in ("RMSE", "NRMSE"):
            score = math.sqrt(score)
            if metric == "NRMSE":
                score = score / (np.amax(data_reference) - np.amin(data_reference))
    elif metric in ("MAE", "NMAE"):
        score = mean_absolute_error(data_reference, data_test)
        if metric == "NMAE":
            score = score / (np.amax(data_reference) - np.amin(data_reference))
    elif metric == "R2":
        score = r2_score(data_reference, data_test, multioutput="variance_weighted")
    else:
        raise ValueError(f"Unknown metric {metric}")

    return score


def _load_simulations(runpath: str, ecl_base: str) -> EclSum:
    """
    Internal helper function to simulation results in parallel.
    Args:
        runpath: Path to where the realization is run.
        ecl_base: Path to where the realization is run.
    Returns:
        EclSum
    """

    return EclSum(str(pathlib.Path(runpath) / pathlib.Path(ecl_base)))


def make_observation_dataframe(obs, key_list_data):
    """
    Internal helper function to generate dataframe containing
    selected observations and respective dates

    Args:
        obs: data read from ERT observation yaml files
        key_list_data: list of selected observations

    Returns:
        Pandas dataframe containing selected observations and
        respective dates
    """

    df_obs = pd.DataFrame()
    for id_key, key in enumerate(key_list_data):
        obs_dates = []
        obs_values = []
        for value in obs.get(key)["observations"]:
            obs_dates.append(np.datetime64(value.get("date")))
            obs_values.append(value.get("value"))
        if id_key == 0:
            df_obs["DATE"] = pd.Series(obs_dates)
        df_obs[key] = pd.Series(obs_values)

    return df_obs


def load_csv_file(csv_file, csv_columns):
    """
    Internal helper function to generate dataframe containing
    selected observations and respective dates

    Args:
        csv_file: name of CSV file
        csv_columns: name of columns of CSV file

    Returns:
        Pandas dataframe containing data from existing CSV file or
        empty dataframe with requested columns if CSV file does not exist
    """

    if os.path.exists(csv_file + ".csv"):
        df = pd.read_csv(csv_file + ".csv")
    else:
        df = pd.DataFrame(columns=csv_columns)

    return df


def compute_metric_ensemble(obs, list_ensembles, metrics, str_key, iteration):
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


def make_dataframe_simulation_data(path, eclbase_file, keys):
    """
    Internal helper function to generate dataframe containing
    data from ensemble of simulations from selected simulation keys

    Args:
        path: path to folder containing ensemble of simulations
        eclbase_file: name of simulation case file
        keys: list of prefix of quantities of interest to be loaded

    Returns:
        df: Pandas dataframe contained data from ensemble of simulations
        realizations_dict: dictionary containing path to loaded simulations
        key_list_data: list of keys corresponding to selected quantities of interest
        iteration: current AHM iteration number
    """

    iteration = int(re.findall(r"[0-9]+", sorted(glob.glob(path))[-1])[-1])
    runpath_list = glob.glob(path[::-1].replace("*", str(iteration), 1)[::-1])

    realizations_dict = {}
    # Load summary files of latest iteration
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    for runpath, eclbase in list(zip(runpath_list, itertools.repeat(eclbase_file))):
        realizations_dict[runpath] = _load_simulations(runpath, eclbase)

    # Prepare dataframe
    df = pd.DataFrame()
    for id_real, runpath in enumerate(realizations_dict.keys()):
        df_tmp = pd.DataFrame()
        dates = realizations_dict[runpath].dates
        if id_real == 0:
            df["DATE"] = pd.Series(dates)
            df["REAL_ID"] = pd.Series(id_real * np.ones(len(dates)), dtype=int)
        df_tmp["DATE"] = pd.Series(dates)
        df_tmp["REAL_ID"] = pd.Series(id_real * np.ones(len(dates)), dtype=int)

        if id_real == 0:
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
            data = realizations_dict[runpath].numpy_vector(key)
            if id_real == 0:
                df[key] = pd.Series(data)
            df_tmp[key] = pd.Series(data)

        if id_real > 0:
            df = df.append(df_tmp, ignore_index=True)

    return df, realizations_dict, key_list_data, iteration


def save_plots_metrics(df_metrics, metrics, str_key):
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

    parser = argparse.ArgumentParser(prog=("Save iteration parameters to a file."))
    parser.add_argument("runpath", type=str, help="Path to the ERT runpath.")
    parser.add_argument(
        "eclbase", type=str, help="Path to the simulation from runpath."
    )
    parser.add_argument("yamlobs", type=str, help="Path to the yaml observation file.")
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
    keys = list(args.quantity.replace("[", "").replace("]", "").split(","))
    metrics = list(args.metrics.replace("[", "").replace("]", "").split(","))

    # Load ensemble of FlowNet
    df, realizations_dict, key_list_data, iteration = make_dataframe_simulation_data(
        args.runpath, args.eclbase, keys
    )

    # Load observation file (OPM reference / truth)
    with open(args.yamlobs) as stream:
        obs = {
            item.pop("key"): item
            for item in yaml.safe_load(stream).get("smry", [dict()])
        }
    df_obs = make_observation_dataframe(obs, key_list_data)

    # Filter dataframe base on measurement dates
    df = df[df["DATE"].isin(df_obs["DATE"])]

    # Compute accuracy over iterations

    # Initiate dataframe with metrics
    df_metrics = load_csv_file(args.outfile, ["quantity", "iteration"] + metrics)

    for str_key in keys:
        truth_data = filter_dataframe(
            df_obs, "DATE", np.datetime64(args.start), np.datetime64(args.end)
        )
        obs_opm = prepare_opm_reference_data(
            truth_data, str_key, len(realizations_dict)
        )

        ens_flownet = []
        ens_flownet.append(
            prepare_flownet_data(
                filter_dataframe(
                    df, "DATE", np.datetime64(args.start), np.datetime64(args.end)
                ),
                str_key,
                len(realizations_dict),
            )
        )

        # Normalizing data
        obs_opm, ens_flownet = normalize_data(obs_opm, ens_flownet)

        # Appending dataframe with accuracy metrics of current iteration
        dict_metric_tmp = compute_metric_ensemble(
            obs_opm, ens_flownet, metrics, str_key, iteration
        )
        df_metrics = df_metrics.append(dict_metric_tmp, ignore_index=True)

        # Plotting accuracy metrics over iterations
        save_plots_metrics(df_metrics, metrics, str_key)

    # Saving accuracy metrics to CSV file
    df_metrics.to_csv(args.outfile + ".csv", index=False)

    print("[Done]")
