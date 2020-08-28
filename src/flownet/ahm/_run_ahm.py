import argparse
from typing import Union, List, Optional

import json
from operator import add

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from configsuite import ConfigSuite

from ..realization import Schedule
from ..network_model import NetworkModel
from ..network_model import create_connections
from ._assisted_history_matching import AssistedHistoryMatching

from ..parameters import (
    PorvPoroTrans,
    RockCompressibility,
    RelativePermeability,
    Aquifer,
    Equilibration,
    FaultTransmissibility,
)
from ..data import FlowData


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


def _find_dist_minmax(min_val: Optional[float], max_val: Optional[float], mean_val: float) -> float:

    """
    Find the distribution min or max for a loguniform distribution, assuming only
    one of these and the mean are given

    Args:
        min_val: minimum value for the distribution
        max_val: maximum value for the distribution
        mean_val: mean value for the distribution

    Returns:
        The missing minimum or maximum value

    """
    # pylint: disable=cell-var-from-loop
    if min_val is None:
        result = minimize(
            lambda x: (mean_val - ((max_val - x) / np.log(max_val / x))) ** 2,
            x0=mean_val,
            tol=1e-9,
            method="L-BFGS-B",
            bounds=[(1e-9, mean_val)],
        ).x[0]
    if max_val is None:
        result = minimize(
            lambda x: (mean_val - ((x - min_val) / np.log(x / min_val))) ** 2,
            x0=mean_val,
            tol=1e-9,
            method="L-BFGS-B",
            bounds=[(mean_val, None)],
        ).x[0]

    return result


def _get_distribution(
    parameters: Union[str, List[str]], parameters_config: dict, index: list
) -> pd.DataFrame:
    """
    Create the distribution min-max for one or more parameters

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

        if parameter_config.mean is not None:
            mean = parameter_config.mean

            if parameter_config.loguniform is True:
                # pylint: disable=cell-var-from-loop
                if parameter_config.max is not None:
                    dist_max = parameter_config.max
                    dist_min = _find_dist_minmax(None, dist_max, mean)
                else:
                    dist_min = parameter_config.min
                    dist_max = _find_dist_minmax(dist_min, None, mean)
            else:
                if parameter_config.max is not None:
                    dist_max = parameter_config.max
                    dist_min = mean - (dist_max - mean)
                else:
                    dist_min = parameter_config.min
                    dist_max = mean + (mean - dist_min)
        else:
            dist_min = parameter_config.min
            dist_max = parameter_config.max

        df[f"minimum_{parameter}"] = dist_min
        df[f"maximum_{parameter}"] = dist_max
        df[f"loguniform_{parameter}"] = parameter_config.loguniform

    return df


def update_distribution(
        parameters: Union[str, List[str]], ahm_case: str
) -> Union[str, List[str]]:
    """
    Update the prior distribution min-max for one or more parameters based on
    the mean of the posterior distribution. It is assumed that the prior min
    and max values cannot be exceeded.

    Args:
        parameters: which parameter(s) should be updated
        ahm_case: path to a previous HM experiment folder

    Returns:
        The parameters list with updated distributions.

    """

    df = pd.read_parquet(ahm_case + "/parameters_iteration-latest.parquet.gzip")

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
            del sorted_random_samples[:n]
            if count_index == 1:
                parameter.mean_values = random_samples
            else:
                parameter.mean_values = list(
                    map(add, parameter.mean_values, random_samples)
                )

    # compute ensemble-mean values
    for parameter in parameters:
        parameter.mean_values = [
            value / float(df.shape[0]) for value in parameter.mean_values
        ]

    # update the distributions
    for parameter in parameters:
        n = len(parameter.random_variables)

        for i, var in enumerate(parameter.random_variables):
            mean = parameter.mean_values[i]

            loguniform = var.ert_gen_kw.split()[0] == "LOGUNIF"
            dist_min0 = var.minimum
            dist_max0 = var.maximum

            if loguniform is True:
                dist_max = dist_max0
                dist_min = _find_dist_minmax(None, dist_max, mean)
                if dist_min < dist_min0:
                    dist_min = dist_min0
                    dist_max = _find_dist_minmax(dist_min, None, mean)
            else:
                dist_max = dist_max0
                dist_min = mean - (dist_max - mean)
                if dist_min < dist_min0:
                    dist_min = dist_min0
                    dist_max = mean + (mean - dist_min)
            var.minimum = dist_min
            var.maximum = dist_max

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
    fast_pyscal = config.flownet.fast_pyscal

    # Load production and well coordinate data
    field_data = FlowData(
        config.flownet.data_source.simulation.input_case,
        perforation_handling_strategy=config.flownet.perforation_handling_strategy,
    )
    df_production_data: pd.DataFrame = field_data.production
    df_coordinates: pd.DataFrame = field_data.coordinates

    # Load fault data if required
    df_fault_planes: Optional[
        pd.DataFrame
    ] = field_data.faults if config.model_parameters.fault_mult else None

    concave_hull_bounding_boxes: Optional[np.ndarray] = None
    if config.flownet.data_source.concave_hull:
        concave_hull_bounding_boxes = field_data.grid_cell_bounding_boxes

    df_connections: pd.DataFrame = create_connections(
        df_coordinates, config, concave_hull_bounding_boxes=concave_hull_bounding_boxes
    )

    network = NetworkModel(
        df_connections,
        cell_length=cell_length,
        area=area,
        fault_planes=df_fault_planes,
        fault_tolerance=config.flownet.fault_tolerance,
    )

    schedule = Schedule(network, df_production_data, config.name)

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

    #########################################
    # Relative Permeability                 #
    #########################################

    # Create a Pandas dataframe with all SATNUMs based on the chosen scheme
    if config.model_parameters.relative_permeability.scheme == "individual":
        df_satnum = pd.DataFrame(
            range(1, len(network.grid.index) + 1), columns=["SATNUM"]
        )
    elif config.model_parameters.relative_permeability.scheme == "global":
        df_satnum = pd.DataFrame([1] * len(network.grid.index), columns=["SATNUM"])
    else:
        raise ValueError(
            f"The relative permeability scheme "
            f"'{config.model_parameters.relative_permeability.scheme}' is not valid.\n"
            f"Valid options are 'global' or 'individual'."
        )

    # Create a pandas dataframe with all parameter definition for each individual tube
    relperm_dist_values = pd.DataFrame(
        columns=["parameter", "minimum", "maximum", "loguniform", "satnum"]
    )

    relperm_dict = {
        key: values
        for key, values in config.model_parameters.relative_permeability._asdict().items()
        if not all(value is None for value in values)
    }

    relperm_parameters = {
        key: relperm_dict[key] for key in relperm_dict if key != "scheme"
    }

    for i in df_satnum["SATNUM"].unique():
        info = [
            relperm_parameters.keys(),
            [relperm_parameters[key].min for key in relperm_parameters],
            [relperm_parameters[key].max for key in relperm_parameters],
            [False] * len(relperm_parameters),
            [i] * len(relperm_parameters),
        ]

        relperm_dist_values = relperm_dist_values.append(
            pd.DataFrame(
                list(map(list, zip(*info))),
                columns=["parameter", "minimum", "maximum", "loguniform", "satnum"],
            ),
            ignore_index=True,
        )

    #########################################
    # Equilibration                         #
    #########################################

    # Create a Pandas dataframe with all EQLNUM based on the chosen scheme
    if config.model_parameters.equil.scheme == "individual":
        df_eqlnum = pd.DataFrame(
            range(1, len(network.grid.index) + 1), columns=["EQLNUM"]
        )
    elif config.model_parameters.equil.scheme == "global":
        df_eqlnum = pd.DataFrame([1] * len(network.grid.index), columns=["EQLNUM"])
    else:
        raise ValueError(
            f"The equilibration scheme "
            f"'{config.model_parameters.relative_permeability.scheme}' is not valid.\n"
            f"Valid options are 'global' or 'individual'."
        )

    # Create a pandas dataframe with all parameter definition for each individual tube
    equil_dist_values = pd.DataFrame(
        columns=["parameter", "minimum", "maximum", "loguniform", "eqlnum"]
    )

    equil_config = config.model_parameters.equil
    for i in df_eqlnum["EQLNUM"].unique():
        info = [
            ["datum_pressure", "owc_depth", "gwc_depth", "goc_depth"],
            [
                equil_config.datum_pressure.min,
                None if equil_config.owc_depth is None else equil_config.owc_depth.min,
                None if equil_config.gwc_depth is None else equil_config.gwc_depth.min,
                None if equil_config.goc_depth is None else equil_config.goc_depth.min,
            ],
            [
                equil_config.datum_pressure.max,
                None if equil_config.owc_depth is None else equil_config.owc_depth.max,
                None if equil_config.gwc_depth is None else equil_config.gwc_depth.max,
                None if equil_config.goc_depth is None else equil_config.goc_depth.max,
            ],
            [False] * 4,
            [i] * 4,
        ]

        equil_dist_values = equil_dist_values.append(
            pd.DataFrame(
                list(map(list, zip(*info))),
                columns=["parameter", "minimum", "maximum", "loguniform", "eqlnum"],
            ),
            ignore_index=True,
        )

    equil_dist_values.dropna(inplace=True)

    #########################################
    # Fault transmissibility                #
    #########################################

    if isinstance(network.faults, dict):
        fault_mult_dist_values = _get_distribution(
            ["fault_mult"], config.model_parameters, list(network.faults.keys()),
        )

    #########################################
    # Aquifer                               #
    #########################################

    if all(config.model_parameters.aquifer) and all(
        config.model_parameters.aquifer.size_in_bulkvolumes
    ):

        aquifer_config = config.model_parameters.aquifer

        # Create a Pandas dataframe with parameters for all aquifers, based on the chosen scheme
        if aquifer_config.scheme == "individual":
            df_aquid = pd.DataFrame(
                range(1, len(network.aquifers_xyz) + 1), columns=["AQUID"]
            )
        elif aquifer_config.scheme == "global":
            df_aquid = pd.DataFrame([1] * len(network.aquifers_xyz), columns=["AQUID"])
        else:
            raise ValueError(
                f"The aquifer scheme "
                f"'{aquifer_config['scheme']}' is not valid.\n"
                f"Valid options are 'global' or 'individual'."
            )

        # Create a pandas dataframe with all parameter definition for each individual tube
        aquifer_dist_values = pd.DataFrame(
            columns=["parameter", "minimum", "maximum", "loguniform", "aquid"]
        )

        aquifer_parameters = {
            key: value
            for key, value in aquifer_config._asdict().items()
            if key not in ("scheme", "type", "fraction", "delta_depth", "datum_depth")
        }

        for i in df_aquid["AQUID"].unique():
            info = [
                aquifer_parameters.keys(),
                [param.min for param in aquifer_parameters.values()],
                [param.max for param in aquifer_parameters.values()],
                [param.loguniform for param in aquifer_parameters.values()],
                [i] * len(aquifer_parameters),
            ]

            aquifer_dist_values = aquifer_dist_values.append(
                pd.DataFrame(
                    list(map(list, zip(*info))),
                    columns=["parameter", "minimum", "maximum", "loguniform", "aquid"],
                ),
                ignore_index=True,
            )

    parameters = [
        PorvPoroTrans(porv_poro_trans_dist_values, ti2ci, network),
        RelativePermeability(
            relperm_dist_values,
            ti2ci,
            df_satnum,
            config.flownet.phases,
            fast_pyscal=fast_pyscal,
        ),
        Equilibration(
            equil_dist_values,
            network,
            ti2ci,
            df_eqlnum,
            equil_config.datum_depth,
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

    if all(config.model_parameters.aquifer) and all(
        config.model_parameters.aquifer.size_in_bulkvolumes
    ):
        parameters.append(
            Aquifer(aquifer_dist_values, network, scheme=aquifer_config.scheme)
        )

    if isinstance(network.faults, dict):
        parameters.append(FaultTransmissibility(fault_mult_dist_values, network))

    if config.model_parameters.ahm_case is not None:
        parameters = update_distribution(parameters, config.model_parameters.ahm_case)

    ahm = AssistedHistoryMatching(network, schedule, parameters, config,)

    ahm.create_ert_setup(
        args=args, training_set_fraction=_find_training_set_fraction(schedule, config),
    )

    ahm.report()

    ahm.run_ert(weights=config.ert.ensemble_weights)
