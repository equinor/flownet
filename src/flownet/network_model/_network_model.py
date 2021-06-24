from typing import List, Tuple, Optional, Dict, Any, Union
from itertools import combinations, repeat, compress

import numpy as np
import pandas as pd
import pyvista as pv
from scipy import spatial

from ._one_dimensional_model import OneDimensionalModel
from ..utils.types import Coordinate
from ..utils.raytracing import moller_trumbore


class NetworkModel:
    def __init__(
        self,
        df_entity_connections: pd.DataFrame,
        df_well_connections: pd.DataFrame,
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
            df_well_connections: A dataframe containing information about the (individual) connections of a
                well, and their open or closed state throughout time.
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
        self.df_well_connections: pd.DataFrame = df_well_connections
        self._cell_length: float = cell_length
        self._area: float = area
        self._grid: pd.DataFrame = self._calculate_grid_corner_points()
        self._nncs: List[Tuple[int, int]] = self._calculate_nncs()

        self._initial_cell_volumes = np.ones((len(self.connection_midpoints), 1))

        self._fault_planes: Optional[pd.DataFrame] = None
        self._faults: Optional[Dict] = None
        if isinstance(fault_planes, pd.DataFrame):
            self._fault_planes = fault_planes
            self._faults = self._calculate_faults(fault_tolerance)

    def get_connection_midpoints(self, i: Optional[int] = None) -> np.ndarray:
        """
        Returns a numpy array with the midpoint of each connection in the network or,
        in case the optional i parameter is specified, the midpoint
        of the i'th connection.

        Args:
            i: specific zero-offset element to calculate the midpoint for

        Returns:
            (Nx3) np.ndarray with connection midpoint coordinates.

        """
        selector: Union[slice, int] = slice(len(self._df_entity_connections.index))

        if i is not None:
            if i > len(self._df_entity_connections.index) or i < 0:
                raise ValueError(
                    f"Optional parameter i is '{i}' but should be between 0 and "
                    f"{len(self._df_entity_connections.index)}."
                )
            if not isinstance(i, int):
                raise TypeError(
                    f"Optional parameter i is of type '{type(i).__name__}' "
                    "but should be an integer."
                )
            selector = i

        coordinates_start = self._df_entity_connections[
            ["xstart", "ystart", "zstart"]
        ].values[selector]
        coordinates_end = self._df_entity_connections[
            [
                "xend",
                "yend",
                "zend",
            ]
        ].values[selector]

        return (coordinates_start + coordinates_end) / 2

    @property
    def initial_cell_volumes(self) -> np.ndarray:
        """Initial cell volume for each grid cell in the flownet.

        Returns:
            An np.ndarray with multiplier for each FlowNet grid cell's volume.
        """
        return self._initial_cell_volumes

    @initial_cell_volumes.setter
    def initial_cell_volumes(self, initial_cell_volumes: np.ndarray):
        """Set the initial cell volume for each grid cell in the flownet.

        Args:
            initial_cell_volumes: Array with initial cell volumes
                for each grid cell in the FlowNet.

        Raises:
            TypeError: When the supplied type is not np.ndarray
            ValueError: When the shape not of (N_gridcells X 1).

        Returns:
            Nothing
        """
        if not isinstance(initial_cell_volumes, np.ndarray):
            raise TypeError("The initial_cell_volumes should be of type np.ndarray.")
        if initial_cell_volumes.shape[0] != len(self.cell_midpoints[0]):
            raise ValueError(
                "The shape of the initial_cell_volumes np.ndarray should "
                f"be {len(self.cell_midpoints[0])} x 1."
            )

        self._initial_cell_volumes = initial_cell_volumes

    @property
    def connection_midpoints(self) -> np.ndarray:
        """
        Returns a numpy array with the midpoint of each connection in the network

        Returns:
            (Nx3) np.ndarray with connection midpoint coordinates.

        """
        coordinates_start = self._df_entity_connections[
            ["xstart", "ystart", "zstart"]
        ].values
        coordinates_end = self._df_entity_connections[
            [
                "xend",
                "yend",
                "zend",
            ]
        ].values

        return (coordinates_start + coordinates_end) / 2

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
    def convexhull(self) -> spatial.ConvexHull:  # pylint: disable=maybe-no-member
        """Return the convex hull of the FlowNet network.

        Returns:
            Convex hull of the network.
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

        return spatial.ConvexHull(points)  # pylint: disable=maybe-no-member

    @property
    def total_bulkvolume(self):
        """
        Returns the bulk volume of the network, i.e. the volume of the convex hull of the
        FlowNet network. The calculation is done by finding
        the convex hull of the network (using the end points of the input tubes)
        and then calculating the volume of this hull.

        Returns:
            Bulk volume of the convex hull.

        """
        return self.convexhull.volume  # pylint: disable=no-member

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

    def _create_connection_groups(self) -> pd.DataFrame:
        """
        Create groups of connections at a node.

            1) The input dataframe is stacked such that all start and
               end points are in the same columns. The information regarding if
               it is a start or end point is saved in an extra column called
               'source'.
            2) The start and end dataframes are concatenated in one dataframe.
            3) Binning of (x,y,z) and casting result into floats
            4) Group rows based on equal (x,y,z) bin assignments.

        Returns:

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

        zdist = 1 if np.isclose(maxz, minz) else maxz - minz

        df_concat = (
            df_concat.apply(
                lambda x: np.rint(
                    np.array((x - minx) * 1000 / (maxx - minx), dtype=np.double)
                )
                if x.name in ["x"]
                else x
            )
            .apply(
                lambda y: np.rint(
                    np.array((y - miny) * 1000 / (maxy - miny), dtype=np.double)
                )
                if y.name in ["y"]
                else y
            )
            .apply(
                lambda z: np.rint(np.array((z - minz) * 1000 / zdist, dtype=np.double))
                if z.name in ["z"]
                else z
            )
            .astype({"x": int, "y": int, "z": int})
        )

        return df_concat.groupby(["x", "y", "z"])

    def _calculate_connections_at_nodes(self) -> List[List[int]]:
        """
        Calculates None-Neighbouring-connections (NNCs) using the following approach:

            1) Get connections at a node
            2) Create a list of tube indices of connected nodes

        Returns:
            List of lists with tube indices connected to a node

        """
        connections_at_nodes: List[List[int]] = []

        for _, df_group in self._create_connection_groups():
            connections_at_nodes.append(df_group.index)

        return connections_at_nodes

    def _calculate_nncs(self) -> List[Tuple[int, int]]:
        """
        Calculates None-Neighbouring-connections (NNCs) using the following approach:

            1) Get connections at a node
            2) Calculate all possible combinations of NNCs for the group and
               iterate over them.
            3) The last step is to look up which grid cell indices the two
               identified NNCs correspond to.

        Returns:
            Listing of tuples of NCC connected zero-offset i-coordinates.

        """
        nncs: List[Tuple[int, int]] = []

        for _, df_group in self._create_connection_groups():
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
    ) -> Optional[Dict[str, List[int]]]:
        """
        Calculates fault definitions using the following approach:

            1) Loop through all faults
            2) Perform a triangulation of all points belonging to a fault plane and store the triangles
            3) For each connection, find all triangles in its bounding box, perform ray tracing using the
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
        dict_fault_keyword: Dict[str, List[int]] = {}
        if self._fault_planes is not None:
            fault_names = self._fault_planes["NAME"].unique().tolist()
        if not fault_names:
            return None

        print("Performing fault ray tracing...", end=" ", flush=True)

        # Gather all triangles for all faults and keep track of fault names
        all_triangles_fault_names: List = []
        all_triangles = np.empty(shape=[0, 9])

        for fault_name in fault_names:
            data = self._fault_planes.loc[self._fault_planes["NAME"] == fault_name][
                ["X", "Y", "Z"]
            ].values

            cloud = pv.PolyData(data)
            surf = cloud.delaunay_2d(tol=fault_tolerance)
            # surf.plot(show_edges=True)
            vertices = surf.points[surf.faces.reshape(-1, 4)[:, 1:4].ravel()]

            triangles = np.array(vertices).reshape(-1, 9)

            all_triangles_fault_names.extend(repeat(fault_name, np.shape(triangles)[0]))
            all_triangles = np.append(all_triangles, triangles, axis=0)
            dict_fault_keyword[fault_name] = []

        # Loop through all connections and select all triangles inside of the bounding box of the connection
        # Perform ray tracing on all resulting triangles.
        for index, row in self._df_entity_connections.iterrows():
            bx1, bx2 = sorted([row["xstart"], row["xend"]])
            by1, by2 = sorted([row["ystart"], row["yend"]])
            bz1, bz2 = sorted([row["zstart"], row["zend"]])

            corner1 = np.array([bx1, by1, bz1])
            corner2 = np.array([bx2, by2, bz2])

            vertex1_in_box = np.all(
                np.logical_and(
                    corner1 <= all_triangles[:, 0:3], all_triangles[:, 0:3] <= corner2
                ),
                axis=1,
            )
            vertex2_in_box = np.all(
                np.logical_and(
                    corner1 <= all_triangles[:, 3:6], all_triangles[:, 3:6] <= corner2
                ),
                axis=1,
            )
            vertex3_in_box = np.all(
                np.logical_and(
                    corner1 <= all_triangles[:, 6:9], all_triangles[:, 6:9] <= corner2
                ),
                axis=1,
            )
            triangle_in_box: np.ndarray = np.any(  # type: ignore
                np.column_stack((vertex1_in_box, vertex2_in_box, vertex3_in_box)),
                axis=1,
            )

            triangles_in_bounding_box = all_triangles[triangle_in_box]
            fault_names_in_bounding_box = list(
                compress(all_triangles_fault_names, triangle_in_box)
            )

            # Loop through all triangles inside of the bounding box and perform ray tracing
            cells_in_fault = []
            for (triangle, fault_name) in list(
                zip(triangles_in_bounding_box, fault_names_in_bounding_box)
            ):

                distance = moller_trumbore(
                    row["xstart"],
                    row["ystart"],
                    row["zstart"],
                    row["xend"],
                    row["yend"],
                    row["zend"],
                    *triangle,
                )

                if distance:
                    indices = self._grid.index[self.active_mask(index)].tolist()
                    tube_cell_index = min(
                        max(0, int(distance * len(indices)) - 1), len(indices) - 2
                    )
                    cell_i_index = indices[tube_cell_index]

                    cells_in_fault.append(cell_i_index)

            if len(cells_in_fault) > 0:
                dict_fault_keyword[fault_name].extend(cells_in_fault)

        # Remove empty and double entries
        for fault_name in fault_names:
            if not dict_fault_keyword[fault_name]:
                dict_fault_keyword.pop(fault_name)
            else:
                dict_fault_keyword[fault_name] = list(
                    set(dict_fault_keyword[fault_name])
                )

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

    def bulk_volume_per_flownet_cell_based_on_tube_length(self) -> np.ndarray:
        """Generate bulk volume per flownet grid cell based on the total length
        of the active cells in each tube in the FlowNet and the convex hull on
        the FlowNet network.

        Args:
            network: FlowNet network instance.

        Returns:
            A list with bulk volumes for each flownet tube cell.
        """
        return (
            self.total_bulkvolume
            * self.grid["cell_length"].values
            / self.grid.loc[self.grid["ACTNUM"] == 1, "cell_length"].sum()
            * self.grid["ACTNUM"].values
        )

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

    @property
    def connection_at_nodes(self) -> List[List[int]]:
        """List is lists with tube indices belonging to a node"""
        return self._calculate_connections_at_nodes()

    @property
    def cell_midpoints(self) -> Tuple[Any, Any, Any]:
        """
        Returns a tuple with the midpoint of each cell in the network

        Returns:
            Tuple with connection midpoint coordinates.
        """
        x_mid = (
            self._grid[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]]
            .mean(axis=1)
            .values
        )
        y_mid = (
            self._grid[["y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"]]
            .mean(axis=1)
            .values
        )
        z_mid = (
            self._grid[["z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7"]]
            .mean(axis=1)
            .values
        )

        return (x_mid, y_mid, z_mid)
