from typing import List, Union
import pathlib

import pandas as pd

from ecl import EclDataType
from ecl.eclfile import EclKW
from ecl.grid import EclGrid


def construct_kw(
    name: str, values: List[Union[int, float]], int_type: bool = False
) -> EclKW:
    """
    This function generates a Flow-type keyword for a given set of values.

    Args:
        name: Name for the EclKW to be generated
        values: List of values to transform to an EclKW
        int_type: Use integer (True) or floating point (False) values

    Returns:
        EclKW instance

    """
    ecl_type = EclDataType.ECL_INT if int_type else EclDataType.ECL_FLOAT

    keyword = EclKW(name, len(values), ecl_type)

    for i, _ in enumerate(values):
        keyword[i] = values[i]

    return keyword


def create_egrid(df_coord: pd.DataFrame, filename: pathlib.Path):
    """
    This function does the following:
      - Takes as input a dataframe with coordinates defining all the grid cells
      - Store it as a Flow .EGRID file called `filename`

    The mandatory dataframe columns are xi, yi, zi (where i is the
    integers 0-7). An optional column name is ACTNUM.

    A grid cell is defined by 8 corner points (4 in the bottom plane, 4 in the
    top plane). The ordering is following the Flow definition):

         2---3           6---7
         |   |           |   |
         0---1           4---5

          j
         /|\
          |
          |
          |
          o---------->  i

    The grid cells are assumed to correspond to one or more one dimensional
    flow models. Between two one dimensional models there should always be
    one inactive cell. Grid cells can be set to be inactive or active using
    0 and 1 respectively in the optional ACTNUM column.

    Args:
        df_coord: Pandas dataframe with coordinates for all grid cells
        filename: Path to the EGRID -file to be stored to disk.

    Returns:
        Nothing

    """
    if "ACTNUM" not in df_coord.columns:
        df_coord["ACTNUM"] = 1

    # See Flow manual for details on input order definition of ZCORN.
    zcorn = (
        df_coord[["z0", "z1"]].values.flatten().tolist()
        + df_coord[["z2", "z3"]].values.flatten().tolist()
        + df_coord[["z4", "z5"]].values.flatten().tolist()
        + df_coord[["z6", "z7"]].values.flatten().tolist()
    )

    # See Flow manual for details on input order definition of COORD.
    coord = (
        df_coord[["x0", "y0", "z0", "x4", "y4", "z4"]].values.flatten().tolist()
        + df_coord.tail(1)[["x1", "y1", "z1", "x5", "y5", "z5"]]
        .values.flatten()
        .tolist()
        + df_coord[["x2", "y2", "z2", "x6", "y6", "z6"]].values.flatten().tolist()
        + df_coord.tail(1)[["x3", "y3", "z3", "x7", "y7", "z7"]]
        .values.flatten()
        .tolist()
    )

    actnum = df_coord["ACTNUM"].astype(int).values.flatten().tolist()

    EclGrid.create(
        (len(df_coord.index), 1, 1),
        construct_kw("ZCORN", zcorn),
        construct_kw("COORD", coord),
        construct_kw("ACTNUM", actnum, int_type=True),
    ).save_EGRID(str(filename))
