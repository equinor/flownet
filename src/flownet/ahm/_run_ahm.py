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
    name: str,
) -> List[int]:
    """
        The function loops through each cell in a flow tube, and checks what 'name' region the
        corresponding position (cell midpoint) in the data source simulation model has. If different
        cells in one flow tube are located in different 'name' regions of the original model, the mode is used.
        If flow tubes are entirely outside of the data source simulation grid,
        the 'name' region closest to the midpoint of the flow tube is used.

    Args:
        network: FlowNet network instance
        field_data: FlowData class with information from simulation model data source
        ti2ci: A dataframe with index equal to tube model index, and one column which equals cell indices.
        name: The same of the region parameter

    Returns:
        A list with values for 'name' region for each tube in the FlowNet model
    """
    df_regions = []

    x_mid = network.grid[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]].mean(axis=1)
    y_mid = network.grid[["y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"]].mean(axis=1)
    z_mid = network.grid[["z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7"]].mean(axis=1)

    for i in network.grid.model.unique():
        tube_regions = []
        for j in ti2ci[ti2ci.index == i].values:
            ijk = field_data.grid.find_cell(x_mid[j], y_mid[j], z_mid[j])
            if ijk is not None and field_data.grid.active(ijk=ijk):
                tube_regions.append(field_data.init(name)[ijk])
        if tube_regions != []:
            df_regions.append(mode(tube_regions).mode.tolist()[0])
        else:
            tube_midpoint = (
                network.df_entity_connections.iloc[i][
                    ["xstart", "ystart", "zstart"]
                ].values
                + network.df_entity_connections.iloc[i][["xend", "yend", "zend"]].values
            ) / 2
            dist_to_cell = []
            for k in range(1, field_data.grid.get_num_active()):
                cell_midpoint = field_data.grid.get_xyz(active_index=k)
                dist_to_cell.append(
                    np.sqrt(
                        np.square(tube_midpoint[0] - cell_midpoint[0])
                        + np.square(tube_midpoint[1] - cell_midpoint[1])
                        + np.square(tube_midpoint[2] - cell_midpoint[2])
                    )
                )
            df_regions.append(
                field_data.init(name)[
                    field_data.grid.get_ijk(
                        active_index=dist_to_cell.index(min(dist_to_cell))
                    )
                ]
            )
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

    if config.model_parameters.equil.scheme == "regions_from_sim":
        equil_config_eqlnum = config.model_parameters.equil.eqlnum_region
    else:
        equil_config_eqlnum = [config.model_parameters.equil.eqlnum_region[0]]

    for i in df_eqlnum["EQLNUM"].unique():
        for reg in equil_config_eqlnum:
            if reg.id == i or reg.id is None:
                info = [
                    ["datum_pressure", "owc_depth", "gwc_depth", "goc_depth"],
                    [
                        reg.datum_pressure.min,
                        None if reg.owc_depth is None else reg.owc_depth.min,
                        None if reg.gwc_depth is None else reg.gwc_depth.min,
                        None if reg.goc_depth is None else reg.goc_depth.min,
                    ],
                    [
                        reg.datum_pressure.max,
                        None if reg.owc_depth is None else reg.owc_depth.max,
                        None if reg.gwc_depth is None else reg.gwc_depth.max,
                        None if reg.goc_depth is None else reg.goc_depth.max,
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

    datum_depths = []
    if config.model_parameters.equil.scheme == "regions_from_sim":
        for reg in equil_config_regions:
            datum_depths.append(reg.datum_depth)
    elif config.model_parameters.equil.scheme == "individual":
        datum_depths = [equil_config_regions[0].datum_depth] * len(
            df_eqlnum["EQLNUM"].unique()
        )
    else:
        datum_depths = [equil_config_regions[0].datum_depth]

    datum_depths = list(datum_depths)

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
