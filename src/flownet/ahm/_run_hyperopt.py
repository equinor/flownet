import argparse
from typing import Optional

import pandas as pd
from configsuite import ConfigSuite
from hyperopt import hp, fmin, tpe

from ..realization import Schedule
from ..network_model import NetworkModel
from ..network_model import create_connections
from ._assisted_history_matching import (
    AssistedHistoryMatching,
    find_training_set_fraction,
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
        network,
        schedule,
        parameters,
        case_name=config.name,
        ert_config=config.ert._asdict(),
        random_seed=config.flownet.random_seed,
    )

    ahm.report()

    # Todo: this sequences needs to be written still

    # define an objective function
    def objective(args):
        # Create ertparam file
        ahm.create_ert_setup(
            args=args, training_set_fraction=find_training_set_fraction(schedule, config),
        )

        # Start simulation
        ahm.run_ert(weights=config.ert.ensemble_weights)

        # Wait.....

        # Extract RMSE
        return 0

    space = hp.choice('a',
                      [
                          ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                          ('case 2', hp.uniform('c2', -10, 10))
                      ])

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
