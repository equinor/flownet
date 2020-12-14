from pathlib import Path

from numpy.testing import assert_almost_equal

from flownet.data.from_flow import FlowData

DATA_FILE = Path.home() / Path("flownet-testdata/norne/input_model/norne/NORNE_ATW2013")


# pylint: disable=protected-access
def test_grid_cell_bounding_boxes() -> None:
    flowdata = FlowData(
        DATA_FILE,
        "multiple_based_on_workovers",
    )

    flowdata._layers = [(1, 22)]
    assert_almost_equal(
        flowdata._grid_cell_bounding_boxes(), flowdata._grid_cell_bounding_boxes(0)
    )

    flowdata._layers = [(1, 2), (3, 4)]
    result = flowdata._grid_cell_bounding_boxes(0)
    assert result.shape[0] == 4455
    assert result.shape[1] == 6

    flowdata._layers = [(1, 2), (3, 22)]
    result = flowdata._grid_cell_bounding_boxes(1)
    assert result.shape[0] == 39976
    assert result.shape[1] == 6

    assert True
