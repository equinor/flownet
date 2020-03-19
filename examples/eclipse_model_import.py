import pandas as pd

from flownet import (
    create_connections,
    NetworkModel,
    SimulationRealization,
    data,
    Schedule,
)
from flownet._utils import write_grdecl_file

# Define variables
area = 100
cell_length = 50
additional_flow_nodes = 100
additional_node_candidates = 1000
random_seed = 123456
hull_factor = 1.1
eclipse_case = "../tests/data/norne/NORNE_ATW2013"

# Load production and well coordinate data
field_data = data.EclipseData(
    eclipse_case, perforation_handling_strategy="bottom_point"
)
df_production_data: pd.DataFrame = field_data.production
df_coordinates: pd.DataFrame = field_data.coordinates

configuration = {
    "flownet": {
        "additional_flow_nodes": additional_flow_nodes,
        "additional_node_candidates": additional_node_candidates,
        "hull_factor": hull_factor,
        "random_seed": random_seed,
    },
    "model_parameters": {"aquifer": None},
}

# Generate connections based on well coordinates
df_connections: pd.DataFrame = create_connections(df_coordinates, configuration)

# Create FlowNet model
network = NetworkModel(df_connections, cell_length=cell_length, area=area)

schedule = Schedule(network, df_production_data)

# Add dynamic variables - one row per active cell
df_dynamic_properties = pd.DataFrame(index=network.grid.index)
df_dynamic_properties["PORO"] = 0.2
df_dynamic_properties["PERMX"] = 1000
df_dynamic_properties["PERMY"] = 1000
df_dynamic_properties["PERMZ"] = 1000

includes = {
    "GRID": write_grdecl_file(df_dynamic_properties, "PERMX")
    + write_grdecl_file(df_dynamic_properties, "PERMY")
    + write_grdecl_file(df_dynamic_properties, "PERMZ")
    + write_grdecl_file(df_dynamic_properties, "PORO")
}

# Setup the simulation model
realization = SimulationRealization(network, schedule, includes)

# Build the model
realization.create_model("./from_eclipse/")
