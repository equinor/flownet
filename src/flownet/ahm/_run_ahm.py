import argparse
from typing import Union, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mode
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
                    dist_min = minimize(
                        lambda x: (mean - ((dist_max - x) / np.log(dist_max / x))) ** 2,
                        x0=mean,
                        tol=1e-9,
                        method="L-BFGS-B",
                        bounds=[(1e-9, mean)],
                    ).x[0]
                else:
                    dist_min = parameter_config.min
                    dist_max = minimize(
                        lambda x: (mean - ((x - dist_min) / np.log(x / dist_min))) ** 2,
                        x0=mean,
                        tol=1e-9,
                        method="L-BFGS-B",
                        bounds=[(mean, None)],
                    ).x[0]
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
    df_fault_planes: Optional[pd.DataFrame] = (
        field_data.faults if config.model_parameters.fault_mult else None
    )

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
            range(1, len(network.grid.model.unique()) + 1), columns=["SATNUM"]
        )
    elif config.model_parameters.relative_permeability.scheme == "regions_from_sim":
        df_satnum = pd.DataFrame(
            _from_regions_to_flow_tubes(network, field_data, ti2ci, "SATNUM"),
            columns=["SATNUM"],
        )
    elif config.model_parameters.relative_permeability.scheme == "global":
        df_satnum = pd.DataFrame(
            [1] * len(network.grid.model.unique()), columns=["SATNUM"]
        )

    # Create a pandas dataframe with all parameter definition for each individual tube
    relperm_dist_values = pd.DataFrame(
        columns=["parameter", "minimum", "maximum", "loguniform", "satnum"]
    )

    relperm_interp_values: Optional[pd.DataFrame] = (
        pd.DataFrame(columns=["parameter", "low", "base", "high", "satnum"])
        if config.model_parameters.relative_permeability.interpolate
        else None
    )

    relperm_parameters = config.model_parameters.relative_permeability.regions[
        0
    ]._asdict()
    relperm_parameters.popitem(last=False)

    relperm_dict = {
        key: values
        for key, values in relperm_parameters.items()
        if not all(value is None for value in values)
    }

    relperm_parameters = {key: relperm_dict[key] for key in relperm_dict}

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
            info = [
                ["interpolate"],
                [-1],
                [1],
                [False],
                [i],
            ]
            interp_info = [
                relperm_parameters.keys(),
                [
                    getattr(relp_config_satnum[idx], key).min
                    for key in relperm_parameters
                ],
                [
                    getattr(relp_config_satnum[idx], key).base
                    for key in relperm_parameters
                ],
                [
                    getattr(relp_config_satnum[idx], key).max
                    for key in relperm_parameters
                ],
                [i] * len(relperm_parameters),
            ]
        else:
            info = [
                relperm_parameters.keys(),
                [
                    getattr(relp_config_satnum[idx], key).min
                    for key in relperm_parameters
                ],
                [
                    getattr(relp_config_satnum[idx], key).max
                    for key in relperm_parameters
                ],
                [False] * len(relperm_parameters),
                [i] * len(relperm_parameters),
            ]

        if relperm_interp_values:
            relperm_interp_values = relperm_interp_values.append(
                pd.DataFrame(
                    list(map(list, zip(*interp_info))),
                    columns=["parameter", "low", "base", "high", "satnum"],
                ),
                ignore_index=True,
            )

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
    equil_dist_values = pd.DataFrame(
        columns=["parameter", "minimum", "maximum", "loguniform", "eqlnum"]
    )

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
        info = [
            ["datum_pressure", "owc_depth", "gwc_depth", "goc_depth"],
            [
                equil_config_eqlnum[idx].datum_pressure.min,
                None
                if equil_config_eqlnum[idx].owc_depth is None
                else equil_config_eqlnum[idx].owc_depth.min,
                None
                if equil_config_eqlnum[idx].gwc_depth is None
                else equil_config_eqlnum[idx].gwc_depth.min,
                None
                if equil_config_eqlnum[idx].goc_depth is None
                else equil_config_eqlnum[idx].goc_depth.min,
            ],
            [
                equil_config_eqlnum[idx].datum_pressure.max,
                None
                if equil_config_eqlnum[idx].owc_depth is None
                else equil_config_eqlnum[idx].owc_depth.max,
                None
                if equil_config_eqlnum[idx].gwc_depth is None
                else equil_config_eqlnum[idx].gwc_depth.max,
                None
                if equil_config_eqlnum[idx].goc_depth is None
                else equil_config_eqlnum[idx].goc_depth.max,
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
            ["fault_mult"],
            config.model_parameters,
            list(network.faults.keys()),
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

    # ******************************************************************************

    datum_depths = list(datum_depths)

    parameters = [
        PorvPoroTrans(porv_poro_trans_dist_values, ti2ci, network),
        RelativePermeability(
            relperm_dist_values,
            ti2ci,
            df_satnum,
            config.flownet.phases,
            interpolation_values=relperm_interp_values,
            fast_pyscal=fast_pyscal,
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

    if all(config.model_parameters.aquifer) and all(
        config.model_parameters.aquifer.size_in_bulkvolumes
    ):
        parameters.append(
            Aquifer(aquifer_dist_values, network, scheme=aquifer_config.scheme)
        )

    if isinstance(network.faults, dict):
        parameters.append(FaultTransmissibility(fault_mult_dist_values, network))

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
