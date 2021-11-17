from typing import List

import pandas as pd

from ecl.grid import EclGrid

from ._coarse_grid import CoarseGrid


class CoarseModel:
    def __init__(
        self, ecl_grid: EclGrid, df_well_connections: pd.DataFrame, partition: List[int]
    ):
        """
        Creates a coarse grid by partitioning the bounding box of the reservoir.
        """
        self._ecl_grid = ecl_grid
        self._df_well_connections: pd.DataFrame = df_well_connections[
            ["X", "Y", "Z"]
        ].to_numpy()
        self._partition = partition
        self._grid: EclGrid = self._create_coarse_grid()

    def _create_coarse_grid(self) -> EclGrid:

        well_coords = self._df_well_connections[["X", "Y", "Z"]].to_numpy()
        CG = CoarseGrid(self._ecl_grid, well_coords, self._partition)
        return CG.create_coarse_grid()
