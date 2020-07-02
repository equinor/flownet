import pathlib

import pandas as pd

from flownet import data
from flownet.network_model import create_connections, NetworkModel
from flownet.realization import SimulationRealization, Schedule
from flownet.utils import write_grdecl_file
from flownet.config_parser import parse_config


CONFIG_FOLDER = pathlib.Path(__file__).resolve().parent / "configs"


def test_flow_import(tmp_path: pathlib.Path) -> None:

    config = parse_config(CONFIG_FOLDER / "norne_parameters.yml")
    model_cross_section_area = 40  # m^2

    # Load production and well coordinate data
    field_data = data.FlowData(
        config.flownet.data_source.input_case,
        perforation_handling_strategy="bottom_point",
    )
    df_production_data: pd.DataFrame = field_data.production
    df_coordinates: pd.DataFrame = field_data.coordinates

    # Generate connections based on well coordinates
    df_connections: pd.DataFrame = create_connections(df_coordinates, config)

    # Create FlowNet model
    network = NetworkModel(
        df_connections,
        cell_length=config.flownet.cell_length,
        area=model_cross_section_area,
    )

    schedule = Schedule(network, df_production_data, config.name)

    # Add dynamic variables - one row per active cell
    df_dynamic_properties = pd.DataFrame(index=network.grid.index)
    df_dynamic_properties["PORO"] = 0.2
    df_dynamic_properties["PERMX"] = 1000
    df_dynamic_properties["PERMY"] = 1000
    df_dynamic_properties["PERMZ"] = 1000

    includes = {
        "GRID": write_grdecl_file(  # type: ignore[operator]
            df_dynamic_properties, "PERMX"
        )
        + write_grdecl_file(df_dynamic_properties, "PERMY")  # type: ignore[operator]
        + write_grdecl_file(df_dynamic_properties, "PERMZ")  # type: ignore[operator]
        + write_grdecl_file(df_dynamic_properties, "PORO")  # type: ignore[operator]
    }

    # Setup the simulation model
    realization = SimulationRealization(network, schedule, includes)

    # Build the model
    realization.create_model(tmp_path)
