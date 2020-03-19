import pathlib

import pandas as pd

import flownet.network_model


def test_single_cell(tmp_path: pathlib.Path) -> None:

    data = [
        {
            "x0": 25,
            "y0": 25,
            "z0": 25,
            "x1": 1,
            "y1": 25,
            "z1": 25,
            "x2": 25,
            "y2": 1,
            "z2": 25,
            "x3": 1,
            "y3": 1,
            "z3": 25,
            "x4": 25,
            "y4": 25,
            "z4": 1,
            "x5": 1,
            "y5": 25,
            "z5": 1,
            "x6": 25,
            "y6": 1,
            "z6": 1,
            "x7": 1,
            "y7": 1,
            "z7": 1,
        }
    ]

    df_coord = pd.DataFrame(data)

    flownet.network_model.create_egrid(df_coord, tmp_path / "SINGLE_CELL.EGRID")
