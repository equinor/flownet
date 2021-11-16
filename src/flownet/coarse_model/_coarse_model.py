from typing import List

import numpy as np
import pandas as pd

import ecl
from ecl.grid import EclGrid

class CoarseModel:
    def __init__(
        self,
        ecl_grid: EclGrid,
        df_well_connections: pd.DataFrame,
        partition: List[int] = [8, 8, 5],
    ):
        """
        Creates a 3D network by partitioning the bounding box of the reservoir.
        """
        self._ecl_grid = ecl_grid
        self._df_well_connections: pd.DataFrame = df_well_connections[["X", "Y", "Z"]].to_numpy()
        self._partition = partition

        self._grid: dict = self._create_coarse_grid()

    def _create_coarse_grid(self) -> dict:

        well_coords = self._df_well_connections[["X", "Y", "Z"]].to_numpy()
        cg = CoarseGrid(self._ecl_grid,
                        well_coords,
                        self._partition)
        return cg.create_coarse_grid()
