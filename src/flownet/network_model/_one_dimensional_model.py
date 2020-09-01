import numpy as np
import pandas as pd

from ..utils.types import Coordinate


class OneDimensionalModel:
    def __init__(
        self,
        start: Coordinate,
        end: Coordinate,
        cell_length: float,
        area: float,
        append_inactive: bool = True,
    ):
        """Creates a set of grid cells between a starting point and an end
        point. These points are placed in the centre of the first and last
        grid cells respectively.

        Args:
            start: A tuple representing start (x, y, z)
            end: A tuple representing end (x, y, z)
            cell_length: Preferred length* of each grid cell along the model.
            area: surface area of the flow path in m^2
            append_inactive: If True, append an extra inactive cell at the end

        * To make start and end actually be the mid points of the first and
          last grid cell, the cell_length will in general only be approximately
          fulfilled. In addition, there will always be created at least two
          grid cells regardless of how large cell_length is.
        """

        self._start_pos = np.array(start).astype(float)
        self._end_pos = np.array(end).astype(float)
        self._append_inactive = append_inactive

        displacement = self._end_pos - self._start_pos
        self._length = np.linalg.norm(displacement)

        self._nactive = int(np.max([2, np.ceil(self._length / cell_length)]))
        self._n = (self._nactive + 1) if self._append_inactive else self._nactive

        # Actual length of a cell after determining the number of grid cells
        self._cell_length_actual = self._length / (self._nactive - 1)

        # Angle between displacement vector and positive z axis (range [0, pi])
        self.angle = np.arccos(np.dot(displacement, [0, 0, 1]) / self._length)

        # Determine if the one dimensional model is completely vertical
        self._vertical = np.isclose(self.angle, 0) or np.isclose(self.angle, np.pi)

        # Move displacement in x direction by 1 m if completely vertical,
        # in order to not break the numerical grid model definition used by
        # e.g. Flow and ResInsight:
        displacement[0] += 1

        # Calculation of perpendicular unit vector (in IJ) plane wrt.
        # projection of displacement vector in IJ plane:
        self.perp_ij_vector = np.array([displacement[1], -displacement[0], 0])
        self.perp_ij_vector /= np.linalg.norm(self.perp_ij_vector)

        # Calculation of unit vector which is perpendicular to both the
        # length direction of the one dimensional model, and the previously
        # calculated perpendicular vector to IJ plane model projection:
        self.down_vector = np.cross(self.perp_ij_vector, displacement / self._length)

        if self.down_vector[2] < 0:  # If pointing upwards, flip it:
            self.down_vector *= -1

        # Calculate grid cell mid points:
        self.mid_points = [
            self._start_pos * (1 - i) + self._end_pos * i
            for i in np.linspace(0, 1, self._nactive)
        ]

        if self._append_inactive:
            self.mid_points.append(2 * self.mid_points[-1] - self.mid_points[-2])

        dx = 0.5 * displacement / (self._nactive - 1)
        dy = 0.5 * np.sqrt(area) * self.perp_ij_vector
        dz = 0.5 * np.sqrt(area) * self.down_vector

        self.data = []

        for mid_point in self.mid_points:

            # We will say that increasing I direction (within libecl
            # terminology) is the same direction as the one dimensional model.

            # We will place the right face (corner points 1-3-5-7), i.e.
            # the face highest up along the I direction, of one grid
            # cell such that it is adjacent to the left face (corner points
            # 0-2-4-6) of the next grid cell.

            corner_number = 0
            grid_cell = {}
            for up_down in [-dz, dz]:
                for left_right in [-dy, dy]:
                    for along_model in [-dx, dx]:
                        corner_pos = mid_point + along_model + left_right + up_down
                        grid_cell[f"x{corner_number}"] = corner_pos[0]
                        grid_cell[f"y{corner_number}"] = corner_pos[1]
                        grid_cell[f"z{corner_number}"] = corner_pos[2]
                        corner_number += 1

            self.data.append(grid_cell)

    @property
    def n(self) -> int:
        """Number of grid cells in model, counting both active and inactive."""
        return self._n

    @property
    def nactive(self) -> int:
        """Number active grid cells in model."""
        return self._nactive

    @property
    def cell_length(self) -> float:
        """Actual cell length of the grid cells."""
        return self._length / self._nactive

    @property
    def start(self) -> np.ndarray:
        """Start position of model (x, y, z) returned as numpy array."""
        return self._start_pos

    @property
    def end(self) -> np.ndarray:
        """End position of model (x, y, z) returned as numpy array."""
        return self._end_pos

    @property
    def length(self) -> float:
        """Length of one dimensional model, measured as distance between
        first and last grid cell mid point.
        """
        return self._length

    @property
    def df_coord(self) -> pd.DataFrame:
        """Returns a dataframe of the grid cells with corresponding coordinates"""
        df_coordinates = pd.DataFrame(self.data)
        df_coordinates["cell_length"] = self._cell_length_actual
        df_coordinates["ACTNUM"] = 1

        if self._append_inactive:
            df_coordinates.at[df_coordinates.index[-1], "ACTNUM"] = 0

        return df_coordinates
