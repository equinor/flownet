import pathlib

import numpy as np
import pandas as pd

from flownet.network_model import OneDimensionalModel, create_egrid


def test_mutliple_one_dimensional(tmp_path: pathlib.Path) -> None:
    cell_length = 5
    cross_section_area = 6  # m^2

    data = np.loadtxt(
        "./three_dimensional_network.csv", delimiter=",", skiprows=1, usecols=range(6)
    )
    df_grid = pd.DataFrame()

    for coordinates in data:
        model = OneDimensionalModel(
            coordinates[:3], coordinates[-3:], cell_length, cross_section_area
        )
        df_grid = df_grid.append(model.df_coord)

    create_egrid(df_grid, tmp_path / "MULTIPLE_ONE_DIMENSIONAL_MODELS.EGRID")
