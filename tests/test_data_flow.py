import glob
import operator
from pathlib import Path

from numpy.testing import assert_almost_equal
from ecl.grid import EclRegion

from flownet.data.from_flow import FlowData

DATA_FILE = Path(glob.glob("../flownet-testdata/*/input_model/*/*.DATA")[0])

# pylint: disable=protected-access
def test_grid_cell_bounding_boxes() -> None:
    layers = ()
    flowdata = FlowData(
        DATA_FILE,
        layers,
        "multiple_based_on_workovers",
    )

    # Test no argument and entire field being equal
    flowdata._layers = (1, flowdata.grid.nz)
    assert_almost_equal(
        flowdata._grid_cell_bounding_boxes(), flowdata._grid_cell_bounding_boxes(0)
    )

    # Test zero'th layer id
    flowdata._layers = ((1, 2), (3, 4))
    result = flowdata._grid_cell_bounding_boxes(0)
    active_cells = EclRegion(flowdata.grid, True)
    active_cells.select_kslice(
        *tuple(map(operator.sub, flowdata._layers[0], (1, 1))), intersect=True
    )
    active_cells.select_active(intersect=True)
    assert result.shape[0] == active_cells.active_size()
    assert result.shape[1] == 6

    # Test last layer id
    flowdata._layers = ((1, 2), (3, 4))
    result = flowdata._grid_cell_bounding_boxes(1)
    active_cells = EclRegion(flowdata.grid, True)
    active_cells.select_kslice(
        *tuple(map(operator.sub, flowdata._layers[-1], (1, 1))), intersect=True
    )
    active_cells.select_active(intersect=True)
    assert result.shape[0] == active_cells.active_size()
    assert result.shape[1] == 6


# def test_well_connections() -> None:
