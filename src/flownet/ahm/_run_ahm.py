import argparse
from typing import Union, List, Optional, Dict
import json
import pathlib
from operator import add
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mode

from configsuite import ConfigSuite

from ..realization import Schedule
from ..network_model import NetworkModel
from ..network_model import create_connections
from ._assisted_history_matching import AssistedHistoryMatching
from ..utils import kriging

from ..parameters import (
    PorvPoroTrans,
    RockCompressibility,
    RelativePermeability,
    Aquifer,
    Equilibration,
    FaultTransmissibility,
    Parameter,
)
from ..data import FlowData


def _from_regions_to_flow_tubes(
    network: NetworkModel,
    field_data: FlowData,
    ti2ci: pd.DataFrame,
    region_name: str,
) -> List[int]:
    """
    The function loops through each cell in all flow tubes, and checks what region the
    corresponding position (cell midpoint) in the data source simulation model has. If different
    cells in one flow tube are located in different regions of the original model, the mode is used.
    If flow tubes are entirely outside of the data source simulation grid, the region of the closest
    flow tube which has already been assigned to a region will be used. The distance between two flow
    tubes is calculated as the distance between the mid points of the two flow tubes.

    Args:
        network: FlowNet network instance
        field_data: FlowData class with information from simulation model data source
        ti2ci: A dataframe with index equal to tube model index, and one column which equals cell indices.
        region_name: The name of the region parameter

    Returns:
        A list with values for 'name' region for each tube in the FlowNet model
    """
    df_regions = []

    xyz_mid = network.cell_midpoints

    tube_outside = []
    for i in network.grid.model.unique():
        tube_regions = []
        for j in ti2ci[ti2ci.index == i].values:
            ijk = field_data.grid.find_cell(xyz_mid[0][j], xyz_mid[1][j], xyz_mid[2][j])
            if ijk is not None and field_data.grid.active(ijk=ijk):
                tube_regions.append(field_data.init(region_name)[ijk])
        if tube_regions:
            df_regions.append(mode(tube_regions).mode.tolist()[0])
        else:
            df_regions.append(None)
            tube_outside.append(i)

    if tube_outside:
        tube_midpoints = network.get_connection_midpoints()
        candidate_tubes = set(network.grid.model.unique()) - set(tube_outside)
        while tube_outside:
            i = tube_outside.pop(0)
            first = True
            for j in candidate_tubes:
                dist_to_tube = np.sqrt(
                    np.square(tube_midpoints[i][0] - tube_midpoints[j][0])
                    + np.square(tube_midpoints[i][1] - tube_midpoints[j][1])
                    + np.square(tube_midpoints[i][2] - tube_midpoints[j][2])
                )
                if first:
                    first = False
                    shortest_dist = dist_to_tube
                    shortest_index = j
                elif dist_to_tube < shortest_dist:
                    shortest_dist = dist_to_tube
                    shortest_index = j

            df_regions[i] = df_regions[shortest_index]
            candidate_tubes.add(i)

    return df_regions


def _find_training_set_fraction(
    schedule: Schedule, config: ConfigSuite.snapshot
) -> float:
    """
    Args:
        schedule: FlowNet Schedule instance
        config: Information from the FlowNet config yaml
    Returns:
        Fraction of the observations to be used as a training set

    """
    training_set_fraction = 1.0

    if config.flownet.training_set_end_date is not None:
        if (
            not schedule.get_first_date()
            <= config.flownet.training_set_end_date
            <= schedule.get_dates()[-1]
        ):
            raise AssertionError("Training set end date outside of date range")
        training_set_fraction = float(
            sum(
                date < config.flownet.training_set_end_date
                for date in schedule.get_dates()
            )
            / len(schedule.get_dates())
        )
    elif config.flownet.training_set_fraction is not None:
        training_set_fraction = config.flownet.training_set_fraction

    return training_set_fraction


def _get_distribution(
    parameters: Union[str, List[str]], parameters_config: dict, index: list
) -> pd.DataFrame:
    """
    Create the distribution min, mean, base, stddev, max for one or more parameters

    Args:
        parameters: which parameter(s) should be outputted in the dataframe
        parameters_config: the parameters definition from the configuration file
        index: listing used to determine how many times to repeat the distribution

    Returns:
        A dataframe with distributions for the requested parameter(s)

    """
    if not isinstance(parameters, list):
        parameters = [parameters]

    df = pd.DataFrame(index=index)

    for parameter in parameters:
        parameter_config = getattr(parameters_config, parameter)
        df[f"minimum_{parameter}"] = parameter_config.min
        df[f"maximum_{parameter}"] = parameter_config.max
        df[f"mean_{parameter}"] = parameter_config.mean
        df[f"base_{parameter}"] = parameter_config.base
        df[f"stddev_{parameter}"] = parameter_config.stddev
        df[f"distribution_{parameter}"] = parameter_config.distribution
    return df


def _constrain_using_well_logs(
    porv_poro_trans_dist_values: pd.DataFrame,
    data: np.ndarray,
    network: NetworkModel,
    measurement_type: str,
    config: ConfigSuite.snapshot,
) -> pd.DataFrame:
    """
    Function to constrain permeability and porosity distributions of flow tubes by using 3D kriging of
    porosity and permeability values from well logs.

    Args:
        porv_poro_trans_dist_values: pre-constraining dataframe
        data: well log data
        network: FlowNet network model
        measurement_type: 'prorosity' or 'permeability' (always log10)
        config: FlowNet configparser snapshot

    Returns:
        Well-log constrained "porv_poro_trans_dist_values" DataFrame

    """
    n = config.flownet.constraining.kriging.n
    n_lags = config.flownet.constraining.kriging.n_lags
    anisotropy_scaling_z = config.flownet.constraining.kriging.anisotropy_scaling_z
    variogram_model = config.flownet.constraining.kriging.variogram_model

    if measurement_type == "permeability":
        data[:, 3] = np.log10(data[:, 3])

    variogram_parameters: Optional[Dict] = None
    if measurement_type == "permeability":
        variogram_parameters = dict(
            config.flownet.constraining.kriging.permeability_variogram_parameters
        )
    elif measurement_type == "porosity":
        variogram_parameters = dict(
            config.flownet.constraining.kriging.porosity_variogram_parameters
        )

    if not variogram_parameters:
        variogram_parameters = None

    k3d3_interpolator, ss3d_interpolator = kriging.execute(
        data,
        n=n,
        n_lags=n_lags,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        anisotropy_scaling_z=anisotropy_scaling_z,
    )

    parameter_min_kriging = k3d3_interpolator(
        network.connection_midpoints
    ) - 2 * np.sqrt(ss3d_interpolator(network.connection_midpoints))

    parameter_max_kriging = k3d3_interpolator(
        network.connection_midpoints
    ) + 2 * np.sqrt(ss3d_interpolator(network.connection_midpoints))

    if measurement_type == "permeability":
        parameter_min_kriging = np.power(10, parameter_min_kriging)
        parameter_max_kriging = np.power(10, parameter_max_kriging)

    parameter_min = np.maximum(
        np.minimum(
            parameter_min_kriging,
            porv_poro_trans_dist_values[f"maximum_{measurement_type}"].values,
        ),
        porv_poro_trans_dist_values[f"minimum_{measurement_type}"].values,
    )
    parameter_max = np.minimum(
        np.maximum(
            parameter_max_kriging,
            porv_poro_trans_dist_values[f"minimum_{measurement_type}"].values,
        ),
        porv_poro_trans_dist_values[f"maximum_{measurement_type}"].values,
    )

    # Set NaN's to the full original range as set in the config
    parameter_min[np.isnan(parameter_min)] = porv_poro_trans_dist_values[
        f"minimum_{measurement_type}"
    ].values[0]
    parameter_max[np.isnan(parameter_max)] = porv_poro_trans_dist_values[
        f"maximum_{measurement_type}"
    ].values[0]

    porv_poro_trans_dist_values[f"minimum_{measurement_type}"] = parameter_min
    porv_poro_trans_dist_values[f"maximum_{measurement_type}"] = parameter_max

    return porv_poro_trans_dist_values


def update_distribution(
    parameters: List[Parameter], ahm_case: pathlib.Path
) -> List[Parameter]:
    """
    Update the prior distribution for one or more parameters based on
    the mean and standard deviation of the posterior distribution. It is assumed that the prior min
    and max values cannot be exceeded. The type of distribution will also be kept.

    For the distributions with a min/max:
        * If the mean value in the posterior is less than in the prior, the minimum value (and the mode) will be kept
           and the maximum value and the mean will be updated.
        * If the mean value in the posterior is larger than in the prior, the maximum value (and the mode) will be
           kept and the minimum value and the mean will be updated.

    For distributions defined by a mean and standard deviation the mean and standard deviation will be updated based
    on the values from the posterior

    Args:
        parameters: which parameter(s) should be updated
        ahm_case: path to a previous HM experiment folder

    Returns:
        The parameters list with updated distributions.

    """

    df = pd.read_parquet(ahm_case.joinpath("parameters_iteration-latest.parquet.gzip"))

    count_index = 0
    for realization_index in df.index.values.tolist():
        count_index += 1
        unsorted_random_samples = json.loads(
            df[df.index == realization_index].transpose().to_json()
        )[str(realization_index)]

        sorted_names = sorted(
            list(unsorted_random_samples.keys()), key=lambda x: int(x.split("_")[0])
        )
        sorted_random_samples = [unsorted_random_samples[name] for name in sorted_names]

        for parameter in parameters:
            n = len(parameter.random_variables)
            random_samples = sorted_random_samples[:n]
            names = sorted_names[:n]
            del sorted_random_samples[:n]
            del sorted_names[:n]
            if count_index == 1:
                parameter.names = names
                parameter.mean_values = random_samples
                parameter.stddev_values = np.power(random_samples, 2)
            else:
                parameter.mean_values = list(
                    map(add, parameter.mean_values, random_samples)
                )
                parameter.stddev_values = list(
                    map(add, parameter.stddev_values, np.power(random_samples, 2))
                )

    # compute ensemble-mean values
    for parameter in parameters:
        parameter.mean_values = [
            value / float(df.shape[0]) for value in parameter.mean_values
        ]
        parameter.stddev_values = np.sqrt(
            [
                value / float(df.shape[0]) - np.power(parameter.mean_values, 2)[i]
                for i, value in enumerate(parameter.stddev_values)
            ]
        )

    # update the distributions
    for parameter in parameters:
        for i, var in enumerate(parameter.random_variables):
            mean = parameter.mean_values[i]
            # if mean in posterior is close to min/max in prior raise warning
            if var.maximum > var.minimum:
                if (mean - var.minimum) / (var.mean - var.minimum) < 0.1 or (
                    var.maximum - mean
                ) / (var.maximum - var.mean) < 0.1:
                    warnings.warn(
                        f"The mean value for the posterior ensemble for {parameter.names[i]} is close to \n"
                        f"the upper or lower bounds in the prior. This will give a very narrow prior range \n"
                        f"in this run. Consider updating before running again. "
                    )
            if var.stddev > 0:
                stddev = parameter.stddev_values[i]
                if stddev / var.stddev < 0.1:
                    warnings.warn(
                        f"The standard deviation for the posterior ensemble for {parameter.names[i]} is much lower \n"
                        f"than the standard deviation in the prior. This will give a very narrow prior range \n"
                        f"in this run. Consider updating before running again. "
                    )

                if mean < var.mean:
                    # Keep the lower limit/minimum and mode for distributions that have those
                    var.update_distribution(
                        mean=mean,
                        stddev=stddev,
                        minimum=var.minimum,
                        mode=var.mode,
                        maximum=None,
                    )
                else:
                    # Keep the upper limit/maximum and mode for distributions that have those
                    var.update_distribution(
                        mean=mean,
                        stddev=stddev,
                        maximum=var.maximum,
                        mode=var.mode,
                        minimum=None,
                    )

    return parameters


# pylint: disable=too-many-branches,too-many-statements
def run_flownet_history_matching(
    config: ConfigSuite.snapshot, args: argparse.Namespace
):
    """
    Creates and runs an ERT setup, given user configuration.

    Args:
        config: Configsuite parsed user provided configuration.
        args: Argparse parsed command line arguments.

    Returns:
        Nothing

    """
    # pylint: disable=too-many-locals

    # Define variables
    area = 100
    cell_length = config.flownet.cell_length
    column_names_probdist = [
        "parameter",
        "minimum",
        "maximum",
        "mean",
        "base",
        "stddev",
        "distribution",
    ]
    # Load production and well coordinate data
    field_data = FlowData(
        config.flownet.data_source.simulation.input_case,
        layers=config.flownet.data_source.simulation.layers,
        perforation_handling_strategy=config.flownet.perforation_handling_strategy,
    )
    df_production_data: pd.DataFrame = field_data.production
    df_well_connections: pd.DataFrame = field_data.well_connections

    # Load log data if required
    df_well_logs: Optional[pd.DataFrame] = (
        field_data.well_logs
        if config.flownet.data_source.simulation.well_logs
        else None
    )

    # Load fault data if required
    df_fault_planes: Optional[pd.DataFrame] = (
        field_data.faults if config.model_parameters.fault_mult else None
    )

    concave_hull_list: Optional[List[np.ndarray]] = None
    if config.flownet.data_source.concave_hull:
        concave_hull_list = []
        for layer_id in df_well_connections["LAYER_ID"].unique():
            concave_hull_list.append(
                field_data.grid_cell_bounding_boxes(layer_id=layer_id)
            )

    df_entity_connections: pd.DataFrame = create_connections(
        df_well_connections[["WELL_NAME", "X", "Y", "Z", "LAYER_ID"]].drop_duplicates(
            keep="first"
        ),
        config,
        concave_hull_list=concave_hull_list,
    )

    network = NetworkModel(
        df_entity_connections=df_entity_connections,
        df_well_connections=df_well_connections,
        cell_length=cell_length,
        area=area,
        fault_planes=df_fault_planes,
        fault_tolerance=config.flownet.fault_tolerance,
        initial_volume_distribution_method=config.flownet.prior_volume_distribution,
        field_data=field_data,
    )

    schedule = Schedule(network, df_production_data, config)

    #########################################
    # Set the range on uncertain parameters #
    #########################################

    ##########################################
    # Pore volume, porosity and permeability #
    ##########################################

    # Create a tube index to cell index dataframe:
    ti2ci = pd.DataFrame(data=network.grid.index, index=network.grid.model)

    porv_poro_trans_dist_values = _get_distribution(
        ["bulkvolume_mult", "porosity", "permeability"],
        config.model_parameters,
        network.grid.model.unique(),
    )

    if df_well_logs is not None and config.flownet.constraining.kriging.enabled:
        # Use well logs to constrain priors.

        perm_data = df_well_logs[["X", "Y", "Z", "PERM"]].values
        poro_data = df_well_logs[["X", "Y", "Z", "PORO"]].values

        porv_poro_trans_dist_values = _constrain_using_well_logs(
            porv_poro_trans_dist_values,
            perm_data,
            network,
            "permeability",
            config=config,
        )
        porv_poro_trans_dist_values = _constrain_using_well_logs(
            porv_poro_trans_dist_values, poro_data, network, "porosity", config=config
        )

    #########################################
    # Relative Permeability                 #
    #########################################

    # Create a Pandas dataframe with all SATNUMs based on the chosen scheme
    if config.model_parameters.relative_permeability.scheme == "individual":
        df_satnum = pd.DataFrame(
            range(1, len(network.grid.model.unique()) + 1), columns=["SATNUM"]
        )
    elif config.model_parameters.relative_permeability.scheme == "regions_from_sim":
        df_satnum = pd.DataFrame(
            _from_regions_to_flow_tubes(network, field_data, ti2ci, "SATNUM"),
            columns=["SATNUM"],
        )
    else:
        df_satnum = pd.DataFrame(
            [1] * len(network.grid.model.unique()), columns=["SATNUM"]
        )

    # Create a pandas dataframe with all parameter definition for each individual tube
    relperm_dist_values = pd.DataFrame(columns=column_names_probdist + ["satnum"])

    relperm_parameters = config.model_parameters.relative_permeability.regions[
        0
    ]._asdict()
    relperm_parameters.pop("id", None)

    relperm_dict = {}
    for key, values in relperm_parameters.items():
        values = values._asdict()
        values.pop("distribution", None)
        values.pop("low_optimistic", None)
        if not all(value is None for _, value in values.items()):
            relperm_dict[key] = values

    relperm_parameters = {key: relperm_dict[key] for key in relperm_dict}

    relperm_interp_values: Optional[pd.DataFrame] = (
        pd.DataFrame(columns=list(relperm_parameters.keys()) + ["CASE", "SATNUM"])
        if config.model_parameters.relative_permeability.interpolate
        else None
    )

    defined_satnum_regions = []
    if config.model_parameters.relative_permeability.scheme == "regions_from_sim":
        relp_config_satnum = config.model_parameters.relative_permeability.regions
        for reg in relp_config_satnum:
            defined_satnum_regions.append(reg.id)
    else:
        relp_config_satnum = [config.model_parameters.relative_permeability.regions[0]]
        defined_satnum_regions.append(None)

    for i in np.sort(df_satnum["SATNUM"].unique()):
        if i in defined_satnum_regions:
            idx = defined_satnum_regions.index(i)
        else:
            idx = defined_satnum_regions.index(None)
        if config.model_parameters.relative_permeability.interpolate:
            interp_info = [
                [
                    getattr(relp_config_satnum[idx], key).min
                    for key in relperm_parameters
                ]
                + ["low"]
                + [i],
                [
                    getattr(relp_config_satnum[idx], key).base
                    for key in relperm_parameters
                ]
                + ["base"]
                + [i],
                [
                    getattr(relp_config_satnum[idx], key).max
                    for key in relperm_parameters
                ]
                + ["high"]
                + [i],
            ]
            low_optimistic = [
                getattr(relp_config_satnum[idx], key).low_optimistic
                for key in relperm_parameters
            ]
            for j, val in enumerate(low_optimistic):
                if val:
                    interp_info[0][j], interp_info[2][j] = (
                        interp_info[2][j],
                        interp_info[0][j],
                    )

            info: List = [
                ["interpolate"],
                [-1],
                [1],
                [None],
                [None],
                [None],
                ["uniform"],
                [i],
            ]
            if {"oil", "gas", "water"}.issubset(config.flownet.phases):
                add_info = ["interpolate gas", -1, 1, None, None, None, "uniform", i]
                for j, val in enumerate(add_info):
                    info[j].append(val)

        else:
            info = [relperm_parameters.keys()]
            for keyword in ["min", "max", "mean", "base", "stddev", "distribution"]:
                info.append(
                    [
                        getattr(getattr(relp_config_satnum[idx], key), keyword)
                        for key in relperm_parameters
                    ]
                )
            info.append([i] * len(relperm_parameters))

        if isinstance(relperm_interp_values, pd.DataFrame):
            relperm_interp_values = relperm_interp_values.append(
                pd.DataFrame(
                    list(map(list, interp_info)),
                    columns=list(relperm_parameters.keys()) + ["CASE", "SATNUM"],
                ),
                ignore_index=True,
            )

        relperm_dist_values = relperm_dist_values.append(
            pd.DataFrame(
                list(map(list, zip(*info))),
                columns=column_names_probdist + ["satnum"],
            ),
            ignore_index=True,
        )

    #########################################
    # Equilibration                         #
    #########################################

    # Create a Pandas dataframe with all EQLNUM based on the chosen scheme
    if config.model_parameters.equil.scheme == "individual":
        df_eqlnum = pd.DataFrame(
            range(1, len(network.grid.model.unique()) + 1), columns=["EQLNUM"]
        )
    elif config.model_parameters.equil.scheme == "regions_from_sim":
        df_eqlnum = pd.DataFrame(
            _from_regions_to_flow_tubes(network, field_data, ti2ci, "EQLNUM"),
            columns=["EQLNUM"],
        )
    elif config.model_parameters.equil.scheme == "global":
        df_eqlnum = pd.DataFrame(
            [1] * len(network.grid.model.unique()), columns=["EQLNUM"]
        )

    # Create a pandas dataframe with all parameter definition for each individual tube
    equil_dist_values = pd.DataFrame(columns=column_names_probdist + ["eqlnum"])

    defined_eqlnum_regions = []
    datum_depths = []
    if config.model_parameters.equil.scheme == "regions_from_sim":
        equil_config_eqlnum = config.model_parameters.equil.regions
        for reg in equil_config_eqlnum:
            defined_eqlnum_regions.append(reg.id)
    else:
        equil_config_eqlnum = [config.model_parameters.equil.regions[0]]
        defined_eqlnum_regions.append(None)

    for i in np.sort(df_eqlnum["EQLNUM"].unique()):
        if i in defined_eqlnum_regions:
            idx = defined_eqlnum_regions.index(i)
        else:
            idx = defined_eqlnum_regions.index(None)
        datum_depths.append(equil_config_eqlnum[idx].datum_depth)
        info = [["datum_pressure", "owc_depth", "gwc_depth", "goc_depth"]]
        for keyword in ["min", "max", "mean", "base", "stddev", "distribution"]:
            info.append(
                [
                    getattr(equil_config_eqlnum[idx].datum_pressure, keyword),
                    None
                    if equil_config_eqlnum[idx].owc_depth is None
                    else getattr(equil_config_eqlnum[idx].owc_depth, keyword),
                    None
                    if equil_config_eqlnum[idx].gwc_depth is None
                    else getattr(equil_config_eqlnum[idx].gwc_depth, keyword),
                    None
                    if equil_config_eqlnum[idx].goc_depth is None
                    else getattr(equil_config_eqlnum[idx].goc_depth, keyword),
                ]
            )
        info.append([i] * 4)

        equil_dist_values = equil_dist_values.append(
            pd.DataFrame(
                list(map(list, zip(*info))),
                columns=column_names_probdist + ["eqlnum"],
            ),
            ignore_index=True,
        )

    equil_dist_values = equil_dist_values[equil_dist_values.isnull().sum(axis=1) < 5]

    #########################################
    # Fault transmissibility                #
    #########################################

    if isinstance(network.faults, dict):
        fault_mult_dist_values = _get_distribution(
            ["fault_mult"],
            config.model_parameters,
            list(network.faults.keys()),
        )

    #########################################
    # Aquifer                               #
    #########################################

    if any(config.model_parameters.aquifer[0:3]):

        aquifer_config = config.model_parameters.aquifer

        # Create a Pandas dataframe with parameters for all aquifers, based on the chosen scheme
        if aquifer_config.scheme == "individual":
            df_aquid = pd.DataFrame(
                range(1, len(network.aquifers_xyz) + 1), columns=["AQUID"]
            )
        elif aquifer_config.scheme == "global":
            df_aquid = pd.DataFrame([1] * len(network.aquifers_xyz), columns=["AQUID"])

        # Create a pandas dataframe with all parameter definition for each individual tube
        aquifer_dist_values = pd.DataFrame(columns=column_names_probdist + ["aquid"])

        aquifer_parameters = {
            key: value
            for key, value in aquifer_config._asdict().items()
            if key not in ("scheme", "type", "fraction", "delta_depth", "datum_depth")
        }

        for i in df_aquid["AQUID"].unique():
            info = [aquifer_parameters.keys()]
            for keyword in ["min", "max", "mean", "base", "stddev", "distribution"]:
                info.append(
                    [getattr(param, keyword) for param in aquifer_parameters.values()],
                )
            info.append([i] * len(aquifer_parameters))

            aquifer_dist_values = aquifer_dist_values.append(
                pd.DataFrame(
                    list(map(list, zip(*info))),
                    columns=column_names_probdist + ["aquid"],
                ),
                ignore_index=True,
            )

    datum_depths = list(datum_depths)

    # ******************************************************************************

    datum_depths = list(datum_depths)

    parameters = [
        PorvPoroTrans(
            porv_poro_trans_dist_values,
            ti2ci,
            network,
            config.flownet.min_permeability,
        ),
        RelativePermeability(
            relperm_dist_values,
            ti2ci,
            df_satnum,
            config,
            interpolation_values=relperm_interp_values,
        ),
        Equilibration(
            equil_dist_values,
            network,
            ti2ci,
            df_eqlnum,
            datum_depths,
            config.flownet.pvt.rsvd,
        ),
    ]

    if all(config.model_parameters.rock_compressibility):
        parameters.append(
            RockCompressibility(
                config.model_parameters.rock_compressibility.reference_pressure,
                config.model_parameters.rock_compressibility.min,
                config.model_parameters.rock_compressibility.max,
            ),
        )

    if all(config.model_parameters.aquifer) and any(
        config.model_parameters.aquifer.size_in_bulkvolumes
    ):
        parameters.append(
            Aquifer(aquifer_dist_values, network, scheme=aquifer_config.scheme)
        )

    if isinstance(network.faults, dict):
        parameters.append(FaultTransmissibility(fault_mult_dist_values, network))

    if hasattr(args, "restart_folder") and args.restart_folder is not None:
        parameters = update_distribution(parameters, args.restart_folder)

    ahm = AssistedHistoryMatching(
        network,
        schedule,
        parameters,
        config,
    )

    ahm.create_ert_setup(
        args=args,
        training_set_fraction=_find_training_set_fraction(schedule, config),
    )

    ahm.report()

    ahm.run_ert(weights=config.ert.ensemble_weights)
