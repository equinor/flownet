from typing import List

import pandas as pd
import numpy as np

from ecl.grid import EclGrid

from ._coarse_grid import CoarseGrid
from ._create_df_grid import create_df_grid


class CoarseModel:
    def __init__(
        self, ecl_grid: EclGrid, df_well_connections: pd.DataFrame, partition: List[int]
    ):
        """
        Creates a coarse grid by partitioning the bounding box of the reservoir.
        """
        self._ecl_grid = ecl_grid
        self._well_coords: np.array = df_well_connections[["X", "Y", "Z"]].to_numpy()
        self._partition = partition
        self._grid: pd.DataFrame = self._create_coarse_grid()

    def _create_coarse_grid(self) -> None:
        """
        Create a coarse grid
        """

        CG = CoarseGrid(self._ecl_grid, self._well_coords, self._partition)

        # Convert to pd matching NetworkModel.grid
        self._grid = create_df_grid(
            CG.coordinates, CG.num_nodes, CG.num_elements, CG.actnum
        )
