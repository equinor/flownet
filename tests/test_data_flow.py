import glob
import operator
from pathlib import Path
import pandas as pd
import numpy as np

from numpy.testing import assert_almost_equal
from ecl.grid import EclRegion

from flownet.data.from_flow import FlowData
from flownet.network_model import NetworkModel


def _locate_test_case() -> Path:
    """
    This function will try to find the test data. On the CI/CD this will be
    any case. Locally, the Norne case will be used. If not found, a
    FileNotFoundError will be raised.

    Raises:
        FileNotFoundError: The test data was not found.

    Returns:
        Path to the .DATA file.
    """
    cicd_str = "./flownet-testdata/*/input_model/*/*.DATA"
    home_path = Path.home() / Path(
        "flownet-testdata/norne/input_model/norne/NORNE_ATW2013.DATA"
    )
    flow_path = (
        Path.cwd() / "../flownet-testdata/norne/input_model/norne/NORNE_ATW2013.DATA"
    )

    if len(glob.glob(cicd_str)) > 0:
        # CI/CD
        data_file = Path(glob.glob(cicd_str)[0])
    elif home_path.exists():
        # Test-data repository located in home directory
        data_file = home_path
    elif flow_path.exists():
        # Test-data repositry located in the same location as the flownet repository
        data_file = flow_path
    else:
        raise FileNotFoundError(
            "To be able to run the tests one needs to clone the flownet-testdata from \
            https://github.com/equinor/flownet-testdata. The test-data needs to be located \
            in your home folder OR in the folder containing the flownet repository."
        )

    return data_file


# pylint: disable=protected-access
def test_grid_cell_bounding_boxes() -> None:
    layers = ()
    flowdata = FlowData(_locate_test_case(), layers)

    # Test one layer for the whole field and no layers equal
    flowdata._layers = ((1, flowdata.grid.nz),)
    field_one_layer = flowdata.grid_cell_bounding_boxes(0)

    flowdata._layers = ()
    field_no_layer = flowdata.grid_cell_bounding_boxes(0)
    assert_almost_equal(field_one_layer, field_no_layer)

    # Test zero'th layer id
    flowdata._layers = ((1, 2), (3, 4))
    result = flowdata.grid_cell_bounding_boxes(0)
    active_cells = EclRegion(flowdata.grid, True)
    active_cells.select_kslice(
        *tuple(map(operator.sub, flowdata._layers[0], (1, 1))), intersect=True
    )
    active_cells.select_active(intersect=True)
    assert result.shape[0] == active_cells.active_size()
    assert result.shape[1] == 6

    # Test last layer id
    flowdata._layers = ((1, 2), (3, 4))
    result = flowdata.grid_cell_bounding_boxes(1)
    active_cells = EclRegion(flowdata.grid, True)
    active_cells.select_kslice(
        *tuple(map(operator.sub, flowdata._layers[-1], (1, 1))), intersect=True
    )
    active_cells.select_active(intersect=True)
    assert result.shape[0] == active_cells.active_size()
    assert result.shape[1] == 6


def test_bulk_volume_per_flownet_cell_based_on_voronoi_of_input_model() -> None:
    df_well_connections = pd.read_csv("./tests/data/df_well_connections.csv")
    df_entity_connections = pd.read_csv("./tests/data/df_entity_connections.csv")

    network = NetworkModel(
        df_entity_connections=df_entity_connections,
        df_well_connections=df_well_connections,
        cell_length=100,
        area=100,
        fault_planes=None,
        fault_tolerance=None,
    )

    layers = ()
    flowdata = FlowData(_locate_test_case(), layers)

    volumes_per_cell = (
        flowdata.bulk_volume_per_flownet_cell_based_on_voronoi_of_input_model(
            network,
        )
    )

    model_cell_volume = [
        (cell.volume * flowdata._init.iget_named_kw("NTG", 0)[cell.active_index])
        for cell in flowdata._grid.cells(active=True)
    ]

    assert np.isclose(sum(volumes_per_cell), sum(model_cell_volume))
