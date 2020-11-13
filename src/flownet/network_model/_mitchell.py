from typing import List, Optional
import time
import logging

from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module
import numpy as np

from ..utils.types import Coordinate


# pylint: disable=too-many-branches,too-many-statements
def mitchell_best_candidate_modified_3d(
    perforations: List[Coordinate],
    num_added_flow_nodes: int,
    num_candidates: int = 1000,
    hull_factor: Optional[float] = None,
    concave_hull_bounding_boxes: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
) -> List[Coordinate]:

    # pylint: disable=too-many-locals,invalid-name
    # Currently 39(!) local variables in this code

    """
    Python implementation of a modified Mitchell's Best-Candidate Algorithm to generate additional flow nodes in
    real field cases.

    The algorithm will generate locations for additional flow nodes where the distance to all existing flow nodes
    is maximized. The added flow nodes are located inside the (possibly scaled) convex hull of the
    supplied perforations.

    Args:
        perforations: Python list of real well coordinate tuples
            [(xr_1, yr_1, zr_1), ..., (xr_N, yr_N, zr_N)]
        num_added_flow_nodes: Number of additional flow nodes to generate
        num_candidates: Number of candidates to consider per additional flow nodes
        hull_factor: Factor to linearly scale the convex hull with. Factor will
            scale the distance of each point from the centroid of all the points.
            When a None is supplied a box-shape is used.
        concave_hull_bounding_boxes: Numpy array with x, y, z min/max boundingboxes for each grid block
        random_seed: Random seed to control the reproducibility of the FlowNet.

    Returns:
        Python list of real/original and added flow node coordinate tuples
            [(xr_1, yr_1, zr_1), ..., (xr_N, yr_N, zr_N), (xi_1, yi1, zi1)
            ... (xi_n_i, yi_n_i, zi_n_i)]

    """
    np.random.seed(random_seed)
    start = time.time()
    logging.info("Adding flow nodes:  0%", end="")

    # Read list of coordinate tuples and convert to 1D-numpy arrays
    x, y, z = (np.asarray(t) for t in zip(*perforations))

    # Number of real wells
    num_points = len(x)

    x_moved = np.zeros(num_points)
    y_moved = np.zeros(num_points)
    z_moved = np.zeros(num_points)

    # Determine whether the complex hull needs to be scaled
    if hull_factor:
        # Calculate the centroid of all real perforations
        centroid = (sum(x) / num_points, sum(y) / num_points, sum(z) / num_points)

        # Loop through all real perforation locations
        for count, point in enumerate(perforations):
            # Linearly scale the well perforation's location relative to the centroid
            moved_points = np.add(hull_factor * np.subtract(point, centroid), centroid)
            x_moved[count] = moved_points[0]
            y_moved[count] = moved_points[1]
            z_moved[count] = moved_points[2]

    if hull_factor:
        x_min = min(x_moved)
        x_max = max(x_moved)
        y_min = min(y_moved)
        y_max = max(y_moved)
        z_min = min(z_moved)
        z_max = max(z_moved)
    else:
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        z_min = min(z)
        z_max = max(z)

    # Determine the convex hull of the original or linearly scaled perforations
    if np.all(i == z[0] for i in z):
        # 2D cases
        hull = Delaunay(np.column_stack([x, y]))
    else:
        # 3D cases
        hull = Delaunay(np.column_stack([x, y, z]))

    # Generate all new flow nodes
    for i in range(num_points, num_points + num_added_flow_nodes):
        mid = time.time()
        if mid - start > 4:
            start = mid
            logging.info(
                f"\rAdding flow nodes:  {int(((i-num_points)/num_added_flow_nodes)*100)}%",
                end="",
            )

        in_hull = np.asarray([False] * num_candidates)
        x_candidate = np.zeros(num_candidates)
        y_candidate = np.zeros(num_candidates)
        z_candidate = np.zeros(num_candidates)

        if concave_hull_bounding_boxes is not None:
            xmin_grid_cells = concave_hull_bounding_boxes[:, 0]
            xmax_grid_cells = concave_hull_bounding_boxes[:, 1]
            ymin_grid_cells = concave_hull_bounding_boxes[:, 2]
            ymax_grid_cells = concave_hull_bounding_boxes[:, 3]
            zmin_grid_cells = concave_hull_bounding_boxes[:, 4]
            zmax_grid_cells = concave_hull_bounding_boxes[:, 5]

        # Repeat while not all random points are inside the convex hull
        while not all(in_hull):
            # Generate a set of random candidates that will be the new
            # flow nodes
            x_candidate_tmp = x_min + np.random.rand(num_candidates) * (x_max - x_min)
            y_candidate_tmp = y_min + np.random.rand(num_candidates) * (y_max - y_min)
            z_candidate_tmp = z_min + np.random.rand(num_candidates) * (z_max - z_min)

            # Update the list of flow node candidates. Only the points that previously
            # were not inside the convex hull are updated.
            np.putmask(x_candidate, np.invert(in_hull), x_candidate_tmp)
            np.putmask(y_candidate, np.invert(in_hull), y_candidate_tmp)
            np.putmask(z_candidate, np.invert(in_hull), z_candidate_tmp)

            candidates = np.vstack([x_candidate, y_candidate, z_candidate]).T

            if concave_hull_bounding_boxes is not None:
                for c_index, candidate in enumerate(candidates):
                    if not in_hull[c_index]:
                        in_hull[c_index] = (
                            (
                                (candidate[0] >= xmin_grid_cells)
                                & (candidate[0] <= xmax_grid_cells)
                            )
                            & (
                                (candidate[1] >= ymin_grid_cells)
                                & (candidate[1] <= ymax_grid_cells)
                            )
                            & (
                                (candidate[2] >= zmin_grid_cells)
                                & (candidate[2] <= zmax_grid_cells)
                            )
                        ).any()

            else:
                # Test whether all points are inside the convex hull of the perforations
                if np.all(i == z[0] for i in z):
                    in_hull = hull.find_simplex(candidates[:, (0, 1)]) >= 0
                else:
                    in_hull = hull.find_simplex(candidates) >= 0

        best_distance = 0
        best_candidate = 0
        # Loop through all generated candidates
        for j in range(0, num_candidates):
            # Calculate the distance (in x,y,z direction, relative to the size of the convex hull) to all points
            # already in the FlowNet
            delta_x_relative = np.power(
                ((x[0:i] - x_candidate[j]) / (x_max - x_min)), 2
            )
            delta_y_relative = np.power(
                ((y[0:i] - y_candidate[j]) / (y_max - y_min)), 2
            )
            delta_z_relative = np.power(
                ((z[0:i] - z_candidate[j]) / (z_max - z_min)), 2
            )
            dists = np.sqrt(delta_x_relative + delta_y_relative + delta_z_relative)

            # Select the shortest distance
            dist = np.min(dists)

            # Check if the shortest is larger than any of the previously
            # generated candidates.
            if dist > best_distance:
                # If the requirement is satisfied, set the current candidate as
                # the best candidate
                best_distance = dist
                best_candidate = j

        # Add the best candidate's coordinates; a new flow node is added
        x = np.append(x, x_candidate[best_candidate])
        y = np.append(y, y_candidate[best_candidate])
        z = np.append(z, z_candidate[best_candidate])

    logging.info("\rAdding flow nodes:  100%\ndone.")

    # Return the real/original and added flow node coordinates as a list of tuples.
    return [(x[i], y[i], z[i]) for i in range(len(x))]
