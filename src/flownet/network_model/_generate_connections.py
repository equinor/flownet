import math
import time
from typing import List, Tuple, Any
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, distance  # pylint: disable=no-name-in-module

from ._mitchell import mitchell_best_candidate_modified_3d
from ..utils.types import Coordinate


def _create_record(
    well_pairs: np.ndarray, dist_matrix: np.ndarray
) -> List[Tuple[Any, ...]]:
    """
    Creates a record of every well_pair triangle for which one of the angles is too large
    (currently set to be above 150 degrees)

    Args:
        well_pairs: All well pairs created through Delaunay triangulation
        dist_matrix: Euclidean distance between well coordinates

    Returns:
         A record of connections (from_well, to_well) of undesirables.

    """
    record: List[Tuple[Any, ...]] = []

    def is_angle_too_large(side_a: float, side_b: float, side_c: float) -> bool:
        """Function checks if there is an angle larger than 150 degrees for a triangle with given side lengths

        Args:
            side_a: Length of side a
            side_b: Length of side b
            side_c: Length of side c

        Returns:
            True if an angle larger than 150 degrees exists

        """
        return (
            np.rad2deg(
                math.acos(
                    (math.pow(side_a, 2) + math.pow(side_b, 2) - math.pow(side_c, 2))
                    / (2 * side_a * side_b)
                )
            )
            > 150
        )

    for vertex_a in range(0, 2):
        for vertex_b in range(vertex_a + 1, 3):
            for vertex_c in range(vertex_b + 1, 4):
                edge_a = dist_matrix[vertex_a, vertex_b]
                edge_b = dist_matrix[vertex_a, vertex_c]
                edge_c = dist_matrix[vertex_b, vertex_c]
                if is_angle_too_large(edge_a, edge_b, edge_c):
                    record.append(tuple(well_pairs[vertex_b, vertex_c]))

                if is_angle_too_large(edge_c, edge_b, edge_a):
                    record.append(tuple(well_pairs[vertex_a, vertex_b]))

                if is_angle_too_large(edge_a, edge_c, edge_b):
                    record.append(tuple(well_pairs[vertex_a, vertex_c]))
    return record


def _remove_recorded_connections(
    record: List[Tuple[Any, ...]], conn_matrix: np.ndarray
) -> np.ndarray:
    """
    This function will take in a list of (i,j)/(j,i) addresses that should be removed = set to zero in
    a given Numpy array.

    Args:
        record: List containing the addresses of elements in conn_matrix that should be removed
        conn_matrix: matrix containing 0s and 1s explaining which nodes should be connected in FlowNet

    Returns:
        Numpy array with recorded elements set to zero.

    """
    for i, j in record:
        conn_matrix[i, j] = conn_matrix[j, i] = 0

    return conn_matrix


def _generate_connections(
    df_coordinates: pd.DataFrame, configuration: Any
) -> Tuple[List[Coordinate], List[Coordinate]]:
    """
    Uses MitchellBestCandidate and Delaunay triangulation to populate the reservoir
    with extra flow nodes and returns the start and end coordinates of all
    connections between wells and extra flow nodes.

    Args:
        df_coordinates: coordinates on original DataFrame format
        configuration: FlowNet configuration yaml

    Returns:
        The tuple consisting of the start_coords and end_coords for connections.

    """
    # pylint: disable=too-many-locals
    # There are currently 28 local variables

    start = time.time()
    well_perforations: List[Coordinate] = df_coordinates[
        ["X", "Y", "Z"]
    ].values.tolist()

    # Generating num_added_flow_nodes additional flow nodes, where each additional node is selected
    # as the one candidate (from num_candidates) with the longest shortest distance to any
    # other existing well (including already added flow nodes)
    coordinates: List[Coordinate] = [
        tuple(elem)
        for elem in mitchell_best_candidate_modified_3d(
            well_perforations,
            num_added_flow_nodes=configuration.flownet.additional_flow_nodes,
            num_candidates=configuration.flownet.additional_node_candidates,
            hull_factor=configuration.flownet.hull_factor,
            random_seed=configuration.flownet.random_seed,
        )
    ]

    print("Generating Delaunay triangulation mesh...", end="")
    # Building a mesh consisting of tetrahedrons where the well
    # perforations and the added fow nodes are the
    # vertices of the tetrahedrons
    triangulation = Delaunay(np.array(coordinates), qhull_options="QJ Pp")
    print("done.")

    def are_points_from_same_existing_entity(point1: Coordinate, point2: Coordinate):
        """
        Helper function to check if two points are originating from the same entity.

        Args:
            point1: coordinate of point 1
            point2: coordinate of point 2

        Returns:
            True if points are coming from the same entity

        """
        entity1 = __get_entity_str(df_coordinates, point1)
        entity2 = __get_entity_str(df_coordinates, point2)
        return (entity1 and entity2) and entity1 == entity2

    print("Creating initial connection matrix and measurements...", end="")
    # If any of the tetrahedrons have an edge connecting well perforations
    # with other well perforations, or with added flow nodes, or between added
    # flow nodes, the connection matrix will be 1 for that "connection pair"
    #
    # conn_matrix will be (num_perfs + num_added_flow_nodes) x (num_perfs + num_added_flow_nodes)
    # well_pairs relates to just one tetrahedron at a time (4x4)
    # dist_matrix will be the Euclidian distance for the pairs in well_pairs
    conn_matrix: np.ndarray = np.zeros((len(coordinates), len(coordinates)))
    well_pairs: np.ndarray = np.full((4, 4), pd.NaT)
    dist_matrix: np.ndarray = np.full((4, 4), np.nan)
    for tetrahedron in range(0, len(triangulation.simplices)):
        for vertex_a in range(0, 3):
            for vertex_b in range(vertex_a + 1, 4):
                flow_node_a = triangulation.simplices[tetrahedron, vertex_a]
                flow_node_b = triangulation.simplices[tetrahedron, vertex_b]

                well_pairs[vertex_a, vertex_b] = [flow_node_a, flow_node_b]
                well_pairs[vertex_b, vertex_a] = [flow_node_b, flow_node_a]

                dist_matrix[vertex_a, vertex_b] = dist_matrix[
                    vertex_b, vertex_a
                ] = distance.euclidean(
                    triangulation.points[flow_node_a], triangulation.points[flow_node_b]
                )

                if are_points_from_same_existing_entity(
                    triangulation.points[flow_node_a], triangulation.points[flow_node_b]
                ):
                    continue

                conn_matrix[flow_node_a, flow_node_b] = conn_matrix[
                    flow_node_b, flow_node_a
                ] = 1
        conn_matrix = _remove_recorded_connections(
            _create_record(well_pairs, dist_matrix), conn_matrix
        )
    print("done.")

    # Are there nodes in the connection matrix that has no connections? Definately a problem
    # it is one of the original well perforations
    print("Checking connectivity of well-perforations...")
    _check_for_none_connectivity_amongst_entities(conn_matrix)

    print("Doing a double-take on potential duplicates...", end="")
    duplicates = []

    for element in set(coordinates):
        if coordinates.count(element) > 1:
            duplicates.append(
                [index for index, value in enumerate(coordinates) if value == element]
            )

    # Use the union of the assigned connections and assign to the pair
    for duplicate in duplicates:
        conn_union = np.zeros(len(conn_matrix[0, :]))

        for i in duplicate:
            conn_union = conn_union + conn_matrix[i, :]

        for i in duplicate:
            conn_matrix[i, :] = conn_union
            conn_matrix[:, i] = conn_union.T
    print("done.")

    end = time.time()

    print(f"Generating connections took: {round(end - start, 4)}s")

    print("Splitting into connection starts and ends...", end="")
    starts: List[Coordinate] = []
    ends: List[Coordinate] = []

    # pylint: disable=E1136
    for i in range(conn_matrix.shape[0]):
        for j in range(i, conn_matrix.shape[1]):
            if conn_matrix[i, j] == 1:
                starts.append(coordinates[i])
                ends.append(coordinates[j])
    print("done.")

    return starts, ends


def _check_for_none_connectivity_amongst_entities(conn_matrix: np.ndarray):
    """
    Prints the number of flow nodes that are not connected to the FlowNet (rows or columns in conn_matrix
    with all zeros), and the total number of connections in the FlowNet (how many ones on either side
    of the diagonal).

    Args:
        conn_matrix: matrix containing 0s and 1s explaining which nodes should be connected in FlowNet

    Returns:
        Nothing

    """
    connections = []
    no_cons = []
    con_less_count = 0
    for node_a, _ in enumerate(conn_matrix):
        if sum(conn_matrix[:, node_a]) == 0:
            con_less_count += 1
            no_cons.append(node_a)
        for node_b in range(node_a, len(conn_matrix)):
            if conn_matrix[node_a][node_b] == 1:
                connections.append((node_a, node_b))
    print(f"{con_less_count} not connected")
    print("no cons:", no_cons)
    print(f"{len(connections)} connections")


def __get_entity_str(df_coordinates: pd.DataFrame, coordinate: Coordinate) -> str:
    """
    Helper function to find the name of the entity, if any, that is located at the specified coordinates.

    Args:
        df_coordinates: Dataframe with entity coordinates
        coordinate: Coordinate to look-up

    Returns:
        Name of the entity or an empty string if no name is connected to the specified coordinates.

    """
    values = df_coordinates.values
    x, y, z = coordinate

    xi = np.where(values[:, 1] == x)[0]
    yi = np.where(values[:, 2] == y)[0]
    zi = np.where(values[:, 3] == z)[0]

    if len(xi) == 1 and len(yi) == 1 and len(zi) == 1:
        # Fast look-up when only a single match was found
        entity = values[xi, 0][0]
    elif len(xi) > 0 and len(yi) > 0 and len(zi) > 0:
        # Slow look-up if multiple matches are found
        entity = df_coordinates.loc[
            (df_coordinates["X"] == x)
            & (df_coordinates["Y"] == y)
            & (df_coordinates["Z"] == z)
        ]["WELL_NAME"].tolist()[0]
    else:
        entity = ""

    return entity


def _create_entity_connection_matrix(
    df_coordinates: pd.DataFrame,
    starts: List[Coordinate],
    ends: List[Coordinate],
    aquifer_starts: List[Coordinate],
    aquifer_ends: List[Coordinate],
) -> pd.DataFrame:
    """
    Converts the the coordinates given for starts and ends to the desired DataFrame format for simulation input.

    Args:
        df_coordinates: original DataFrame version of coordinates
        starts: List of coordinates for all the starting entities
        ends: List of coordinates of all the end entities
        aquifer_starts: List of coordinates for all aquifer starts
        aquifer_ends: List of coordinates of all aquifer ends

    Returns:
        Connection coordinate DataFrame on eclipse desired format.

    """
    print("Creating entity connection DataFrame...", end="")
    columns = [
        "xstart",
        "ystart",
        "zstart",
        "xend",
        "yend",
        "zend",
        "start_entity",
        "end_entity",
    ]
    df_out = pd.DataFrame(columns=columns)

    for start, end in zip(starts, ends):
        str_start_entity = __get_entity_str(df_coordinates, start)
        str_end_entity = __get_entity_str(df_coordinates, end)

        df_out = df_out.append(
            {
                "xstart": start[0],
                "ystart": start[1],
                "zstart": start[2],
                "xend": end[0],
                "yend": end[1],
                "zend": end[2],
                "start_entity": str_start_entity,
                "end_entity": str_end_entity,
            },
            ignore_index=True,
        )

    for start, end in zip(aquifer_starts, aquifer_ends):
        str_start_entity = __get_entity_str(df_coordinates, start)

        df_out = df_out.append(
            {
                "xstart": start[0],
                "ystart": start[1],
                "zstart": start[2],
                "xend": end[0],
                "yend": end[1],
                "zend": end[2],
                "start_entity": str_start_entity,
                "end_entity": "aquifer",
            },
            ignore_index=True,
        )

    print("done.")
    return df_out


def _generate_aquifer_connections(
    starts: List[Coordinate],
    ends: List[Coordinate],
    scheme: str,
    fraction: float = 0.1,
    delta_depth: float = 200,
) -> Tuple[List[Coordinate], List[Coordinate]]:
    """Helper function to create aquifer connections

    Args:
        starts: Flowtube starting coordinates
        ends: Flowtube ending coordinates
        scheme: Aquifer scheme to use ('global' or 'individual')
        fraction: Fraction of deepest nodes to connection to an aquifer
        delta_depth: depth difference between aquifer and connected node

    Returns:
        lists of aquifer starting and ending coordinates.
    """

    print("Adding aquifer nodes...", end="")

    all_coordinates = list(set(starts + ends))
    all_coordinates_sorted = sorted(
        all_coordinates, key=lambda key: key[-1], reverse=True
    )
    aquifer_starts = all_coordinates_sorted[
        0 : math.floor(fraction * len(all_coordinates_sorted))
    ]

    if scheme == "individual":
        aquifer_ends = [
            (
                start_coordinate[0],
                start_coordinate[1],
                start_coordinate[2] + delta_depth,
            )
            for start_coordinate in aquifer_starts
        ]
    elif scheme == "global":
        mid_x = (
            max(all_coordinates, key=itemgetter(0))[0]
            - min(all_coordinates, key=itemgetter(0))[0]
        ) / 2
        mid_y = (
            max(all_coordinates, key=itemgetter(1))[1]
            - min(all_coordinates, key=itemgetter(1))[1]
        ) / 2
        max_z = max(all_coordinates, key=itemgetter(1))[2]

        aquifer_ends = [(mid_x, mid_y, max_z + delta_depth)] * len(aquifer_starts)
    else:
        raise NotImplementedError(
            f"Aquifer scheme '{scheme}' is not supported. Possible schemes are 'global' or 'individual'"
        )

    print("done.")

    return aquifer_starts, aquifer_ends  # type: ignore[return-value]


def create_connections(
    df_coordinates: pd.DataFrame, configuration: Any
) -> pd.DataFrame:
    """
    Creates additional flow nodes to increase complexity of field simulation structure so that history-matching can
    be performed.

    Takes in field-entity-coordinates and desired added complexity arguments and returns eclipse format coordinate
    DataFrame.

    Args:
        df_coordinates: Original structure of entity and X, Y, Z coords
        configuration: FlowNet configuration yaml as dictionary

    Returns:
        Desired restructuring of start-end coordinates into separate columns, as per eclipse needs.

    """
    starts, ends = _generate_connections(df_coordinates, configuration)
    aquifer_starts: List[Coordinate] = []
    aquifer_ends: List[Coordinate] = []

    aquifer_config = configuration.model_parameters.aquifer
    if aquifer_config is not None:
        scheme = aquifer_config.scheme
        fraction = aquifer_config.fraction
        delta_depth = aquifer_config.delta_depth

        aquifer_starts, aquifer_ends = _generate_aquifer_connections(
            starts, ends, scheme, fraction=fraction, delta_depth=delta_depth
        )

    return _create_entity_connection_matrix(
        df_coordinates, starts, ends, aquifer_starts, aquifer_ends
    )
