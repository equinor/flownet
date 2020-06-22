import argparse
import pathlib
from typing import Optional

import pandas as pd
import hyperopt
from configsuite import ConfigSuite

from ..realization import Schedule
from ..network_model import NetworkModel
from ..network_model import create_connections
from ._assisted_history_matching import (
    AssistedHistoryMatching,
    create_parameter_distributions,
)

from ..data import EclipseData


def run_hyperopt(config: ConfigSuite.snapshot, args: argparse.Namespace):
    """
    Creates and runs an ERT setup, given user configuration.

    Args:
        config: Configsuite parsed user provided configuration.
        args: Argparse parsed command line arguments.

    Returns:
        Nothing

    """
    # Define variables
    area = 100
    cell_length = config.flownet.cell_length

    # Load production and well coordinate data
    field_data = EclipseData(
        config.flownet.data_source.eclipse_case,
        perforation_handling_strategy=config.flownet.perforation_handling_strategy,
        resample=config.flownet.data_source.resample,
    )
    df_production_data: pd.DataFrame = field_data.production
    df_coordinates: pd.DataFrame = field_data.coordinates

    # Load fault data if required
    df_fault_planes: Optional[
        pd.DataFrame
    ] = field_data.faults if config.model_parameters.fault_mult else None

    df_connections: pd.DataFrame = create_connections(df_coordinates, config)

    network = NetworkModel(
        df_connections,
        cell_length=cell_length,
        area=area,
        fault_planes=df_fault_planes,
        fault_tolerance=config.flownet.fault_tolerance,
    )

    schedule = Schedule(network, df_production_data, config.name)

    parameters = create_parameter_distributions(config, network)

    ahm = AssistedHistoryMatching(
        config,
        network,
        schedule,
        parameters,
        case_name=config.name,
        random_seed=config.flownet.random_seed,
    )

    ahm.report()

    # Define hyperopt search space
    space = []
    index = 0
    for parameter in parameters:
        for random_variable in parameter.random_variables:
            variable_name = str(index) + "_" + parameter.__class__.__name__.lower()
            index += 1
            space.append(random_variable.hyperopt_distribution(variable_name))

    output_folder = pathlib.Path(args.output_folder)

    ahm.create_ert_setup(args=args)

    # define an objective function
    def objective(x):
        # Create ertparam file
        with open(output_folder / "parameters.ertparam", "w") as fh:
            index = 0  # Prepend index in parameter name to enforce uniqueness in ERT
            for parameter in parameters:
                for _ in parameter.random_variables:
                    variable_name = (
                        str(index) + "_" + parameter.__class__.__name__.lower()
                    )
                    fh.write(f"{variable_name} CONST {x[index]}\n")

                    index += 1

        # pylint: disable=broad-except
        try:
            # Start simulation
            ahm.run_ert(weights=config.ert.ensemble_weights)

            return {
                "loss": 1,
                "status": hyperopt.STATUS_OK,
            }
        except (ValueError, Exception):
            return {"status": hyperopt.STATUS_FAIL}

    # minimize the objective over the space
    hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=100)
