from typing import List, Tuple, Optional, Dict, Any
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.spatial
import pyvista as pv
from tqdm import tqdm

from ._one_dimensional_model import OneDimensionalModel
from ..utils.types import Coordinate
from ..utils.raytracing import moller_trumbore


class NetworkModel:
    def __init__(
        self,
        df_entity_connections: pd.DataFrame,
        cell_length: float,
        area: float,
        fault_planes: Optional[pd.DataFrame] = None,
        fault_tolerance: float = 1.0e-5,
    ):
        """
        Creates a network of one dimensional models.

        Args:
            df_entity_connections: A dataframe containing information about the individual one
                                   dimensional models. Columns required are xstart, ystart, zstart,
                                   xend, yend, zend, start_entity and end_entity
            cell_length: Preferred length of each grid cell along the model.
            area: surface area of the flow path.
            fault_planes: DataFrame with fault plane coordinates
            fault_tolerance: minimum distance between corners of a triangle. This value should be set as low
                             as possible to ensure a high resolution fault plane generation. However, this might lead
                             to a very slow fault tracing process therefore one might want to increase the tolerance.
                             Always check that the resulting lower resolution fault plane still is what you expected.

        Start and end coordinates in df and cell_length should have the same
        unit. The area should have the same unit (but squared) as the
        coordinates and cell_length.

        """
        self._df_entity_connections: pd.DataFrame = df_entity_connections
        self._cell_length: float = cell_length
        self._area: float = area
        self._grid: pd.DataFrame = self._calculate_grid_corner_points()
        self._nncs: List[Tuple[int, int]] = self._calculate_nncs()

        if isinstance(fault_planes, pd.DataFrame):
            self._fault_planes: pd.DataFrame = fault_planes
            self._faults: Optional[Dict] = self._calculate_faults(fault_tolerance)

    @property
    def aquifers_xyz(self) -> List[Coordinate]:
        """
        Returns a list of zero-offset i,j,k coordinates for all aquifers in the model

        Returns:
            List of coordinates for the aquifers

        """
        return self._df_entity_connections.loc[
            self._df_entity_connections["end_entity"] == "aquifer"
        ]

    @property
    def aquifers_i(self) -> List[List[int]]:
        """
        Property that returns a list of list of zero-offset i coordinates for all aquifers in the model.

        Returns:
            List of lists of zero-offset i's of aquifers.

        """
        return self._get_aquifer_i()

    @property
    def total_bulkvolume(self):
        """
        Returns the bulk volume of the network. The calculation is done by finding
        the convex hull of the network (using the end points of the input tubes)
        and then calculating the volume of this hull.

        Returns:
            Bulk volume of the convex hull.

        """
        x = (
            self._df_entity_connections["xstart"].tolist()
            + self._df_entity_connections["xend"].tolist()
        )
        y = (
            self._df_entity_connections["ystart"].tolist()
            + self._df_entity_connections["yend"].tolist()
        )
        z = (
            self._df_entity_connections["zstart"].tolist()
            + self._df_entity_connections["zend"].tolist()
        )

        points = np.array((x, y, z), dtype=float).transpose()

        return scipy.spatial.ConvexHull(points).volume  # pylint: disable=no-member

    def _get_aquifer_i(self) -> List[List[int]]:
        """Helper function to get the aquifer i's with zero-offset.

        Returns:
            List of lists of zero-offset i's of aquifers.

        """
        return [
            self.grid.index[self.active_mask(idx)].tolist()
            for idx in self._df_entity_connections.loc[
                self._df_entity_connections["end_entity"] == "aquifer"
            ].index.values
        ]

    def _calculate_nncs(self) -> List[Tuple[int, int]]:
        """
        Calculates None-Neighbouring-connections (NNCs) using the following approach:

            1) The input dataframe is stacked such that all start and
               end points are in the same columns. The information regarding if
               it is a start or end point is saved in an extra column called
               'source'.
            2) The start and end dataframes are concatenated in one dataframe.
            3) Binning of (x,y,z) and casting result into floats
            4) Group rows based on equal (x,y,z) bin assignments.
            5) Calculate all possible combinations of NNCs for the group and
               iterate over them.
            6) The last step is to look up which grid cell indices the two
               identified NNCs correspond to.

        Returns:
            Listing of tuples of NCC connected zero-offset i-coordinates.

        """
        start_df = self._df_entity_connections[["xstart", "ystart", "zstart"]].rename(
            columns={"xstart": "x", "ystart": "y", "zstart": "z"}
        )
        start_df["source"] = "start"

        # Remove connections to aquifers as they should not be connected with
        # NNCs, even if they visually overlap in the global scheme.
        end_df = self._df_entity_connections.loc[
            self._df_entity_connections["end_entity"] != "aquifer"
        ][["xend", "yend", "zend"]].rename(
            columns={"xend": "x", "yend": "y", "zend": "z"}
        )
        end_df["source"] = "end"

        df_concat = pd.concat([start_df, end_df]).astype(
            {"x": float, "y": float, "z": float}
        )

        minx, miny, minz = df_concat[["x", "y", "z"]].min()
        maxx, maxy, maxz = df_concat[["x", "y", "z"]].max()

        df_concat = (
            df_concat.apply(
                lambda x: np.rint((x - minx) * 1000 / (maxx - minx))
                if x.name in ["x"]
                else x
            )
            .apply(
                lambda y: np.rint((y - miny) * 1000 / (maxy - miny))
                if y.name in ["y"]
                else y
            )
            .apply(
                lambda z: np.rint((z - minz) * 1000 / (maxz - minz))
                if z.name in ["z"]
                else z
            )
            .astype({"x": int, "y": int, "z": int})
        )

        nncs: List[Tuple[int, int]] = []

        for _, df_group in df_concat.groupby(["x", "y", "z"]):
            connections_to_node = df_group.index
            nnc_combinations = combinations(connections_to_node, 2)
            for nnc_combination in nnc_combinations:
                nnc: List[int] = []
                for index in nnc_combination:
                    source = df_group["source"].loc[index]
                    indices = self._grid.index[self.active_mask(index)].tolist()
                    nnc.append(int(indices[0] if source == "start" else indices[-1]))

                nncs.append((nnc[0], nnc[1]))

        return nncs

    # pylint: disable=too-many-locals
    def _calculate_faults(
        self, fault_tolerance: float = 1.0e-05
    ) -> Optional[Dict[Any, List[int]]]:
        """
        Calculates fault definitions using the following approach:

            1) Loop through all faults
            2) Perform a triangulation of all points belonging to a fault
            3) For each triangle, perform ray tracing using the
               Möller-Trumbore intersection algorithm.
            4) If an intersection is found, identify the grid blocks that are
               associated with the intersection.

        Args:
            fault_tolerance: minimum distance between corners of a triangle. This value should be set as low
                             as possible to ensure a high resolution fault plane generation. However, this might lead
                             to a very slow fault tracing process therefore one might want to increase the tolerance.
                             Always check that the resulting lower resolution fault plane still is what you expected.

        Returns:
            Listing of tuples of FAULTS entries with zero-offset i-coordinates, or None if no faults are present.

        """

        dict_fault_keyword = {}

        fault_names = self._fault_planes["NAME"].unique().tolist()

        if not fault_names:
            return None

        print("Performing fault ray tracing...")

        for fault_name in list(tqdm(fault_names, total=len(fault_names))):

            data = self._fault_planes.loc[self._fault_planes["NAME"] == fault_name][
                ["X", "Y", "Z"]
            ].values

            cloud = pv.PolyData(data)
            surf = cloud.delaunay_2d(tol=fault_tolerance)
            # surf.plot(show_edges=True)
            vertices = surf.points[surf.faces.reshape(-1, 4)[:, 1:4].ravel()]

            triangles = np.array(vertices).reshape(-1, 9)
            connections = np.hstack(
                (
                    np.arange(len(self._df_entity_connections)).reshape(-1, 1),
                    self._df_entity_connections[
                        ["xstart", "ystart", "zstart", "xend", "yend", "zend"]
                    ].values,
                )
            )

            combinations_triangles = np.array(triangles).repeat(
                len(connections), axis=0
            )
            connections_connections = np.tile(connections, (len(triangles), 1))

            parameters = list(
                map(tuple, np.hstack((connections_connections, combinations_triangles)))
            )

            cells_in_fault = []

            for row in list(tqdm(parameters, total=len(parameters), desc=fault_name)):
                distance, index = moller_trumbore(*row)

                if distance is not False:
                    indices = self._grid.index[self.active_mask(index)].tolist()
                    tube_cell_index = min(
                        max(0, int(distance * len(indices)) - 1), len(indices) - 2
                    )
                    cell_i_index = indices[tube_cell_index]

                    cells_in_fault.append(cell_i_index)

            if len(cells_in_fault) > 0:
                dict_fault_keyword[fault_name] = list(set(cells_in_fault))

        print("done.")

        return dict_fault_keyword

    def _calculate_grid_corner_points(self) -> pd.DataFrame:
        """
        Reads the FlowNet connections, generates the 1D-models and returns
        a DataFrame with grid corner points.

        Returns:
            DataFrame of all corner-points needed to generate the grid.

        """
        df_grid = pd.DataFrame()

        for index, row in self._df_entity_connections.iterrows():
            start = row[["xstart", "ystart", "zstart"]]
            end = row[["xend", "yend", "zend"]]
            model = OneDimensionalModel(start, end, self._cell_length, self._area)
            new_df = model.df_coord
            new_df["model"] = index
            df_grid = df_grid.append(new_df, sort=False)

        df_grid.reset_index(inplace=True, drop=True)

        df_grid["z_mean"] = df_grid[
            ["z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7"]
        ].mean(axis=1)

        return df_grid

    def active_mask(self, model_index) -> List[bool]:
        """
        Helper function to get out active grid cell indices of
        the one dimensional model in the network with index model_index.

        Args:
            model_index: zero-offset model index

        Returns:
            booleans representing the active cells in the model with model index `model_index`. Again, with zero-offset.

        """
        return (self._grid["model"] == model_index) & (self._grid["ACTNUM"] == 1)

    @property
    def faults(self) -> Optional[Dict[Any, Any]]:
        """Dictionary of fault names containing a list of integer zero-offset I-coordinates belonging to a fault"""
        return self._faults

    @property
    def nncs(self) -> List[Tuple[int, int]]:
        """List of tuples of NNC I-coordinates of connections"""
        return self._nncs

    @property
    def grid(self) -> pd.DataFrame:
        """DataFrame of all corner-points needed to generate the grid"""
        return self._grid

    @property
    def cell_length(self) -> float:
        """Desired length of all cells"""
        return self._cell_length

    @property
    def area(self) -> float:
        """surface area between to grid cells"""
        return self._area

    @property
    def df_entity_connections(self) -> pd.DataFrame:
        """DataFrame containing all connections between entities"""
        return self._df_entity_connections
