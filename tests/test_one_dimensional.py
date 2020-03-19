import pathlib

from flownet.network_model import OneDimensionalModel, create_egrid


def test_one_dimensional(tmp_path: pathlib.Path) -> None:
    """Single one dimensional flow Model example
    """

    n_grid_cells = 10
    model_cross_section_area = 40  # m^2

    start = (5, 2, 1.25)
    end = (50, 50, 50)

    model = OneDimensionalModel(start, end, n_grid_cells, model_cross_section_area)

    create_egrid(model.df_coord, tmp_path / "SINGLE_ONE_DIMENSIONAL.EGRID")
