import glob
import operator
from pathlib import Path

from numpy.testing import assert_almost_equal
from ecl.grid import EclRegion

from flownet.data.from_flow import FlowData


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
        DATA_FILE = Path(glob.glob(cicd_str)[0])
    elif home_path.exists():
        # Test-data repository located in home directory
        DATA_FILE = home_path
    elif flow_path.exists():
        # Test-data repositry located in the same location as the flownet repository
        DATA_FILE = flow_path
    else:
        raise FileNotFoundError(
            "To be able to run the tests one needs to clone the flownet-testdata from \
            https://github.com/equinor/flownet-testdata. The test-data needs to be located \
            in your home folder OR in the folder containing the flownet repository."
        )

    return DATA_FILE


# pylint: disable=protected-access
def test_grid_cell_bounding_boxes() -> None:
    flowdata = FlowData(
        _locate_test_case(),
        "multiple_based_on_workovers",
    )

    # Test no argument and entire field being equal
    flowdata._layers = [(1, flowdata.grid.nz)]
    assert_almost_equal(
        flowdata._grid_cell_bounding_boxes(), flowdata._grid_cell_bounding_boxes(0)
    )

    # Test zero'th layer id
    flowdata._layers = [(1, 2), (3, 4)]
    result = flowdata._grid_cell_bounding_boxes(0)
    active_cells = EclRegion(flowdata.grid, True)
    active_cells.select_kslice(
        *tuple(map(operator.sub, flowdata._layers[0], (1, 1))), intersect=True
    )
    active_cells.select_active(intersect=True)
    assert result.shape[0] == active_cells.active_size()
    assert result.shape[1] == 6

    # Test last layer id
    flowdata._layers = [(1, 2), (3, 4)]
    result = flowdata._grid_cell_bounding_boxes(1)
    active_cells = EclRegion(flowdata.grid, True)
    active_cells.select_kslice(
        *tuple(map(operator.sub, flowdata._layers[-1], (1, 1))), intersect=True
    )
    active_cells.select_active(intersect=True)
    assert result.shape[0] == active_cells.active_size()
    assert result.shape[1] == 6
