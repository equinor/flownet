import numpy as np


def check_in_hull(
    concave_hull_bounding_boxes: np.ndarray,
    coordinates: np.ndarray,
    in_hull_known: np.ndarray = None,
) -> np.ndarray:
    """Checks if all coordinates are inside the concave hull bounding boxes.

    Args:
        concave_hull_bounding_boxes: Bounding boxes for the concave hull
        connections: All coordianates to check
        in_hull_known: List of booleans indicating whether to skip a coordinate
            if it is already known to be inside the hull.

    Returns:
        np.ndarray with bools
    """

    xmin_grid_cells = concave_hull_bounding_boxes[:, 0]
    xmax_grid_cells = concave_hull_bounding_boxes[:, 1]
    ymin_grid_cells = concave_hull_bounding_boxes[:, 2]
    ymax_grid_cells = concave_hull_bounding_boxes[:, 3]
    zmin_grid_cells = concave_hull_bounding_boxes[:, 4]
    zmax_grid_cells = concave_hull_bounding_boxes[:, 5]

    if in_hull_known is not None:
        in_hull = in_hull_known
    else:
        in_hull = np.asarray([False] * coordinates.shape[0])

    for c_index, coordinate in enumerate(coordinates):
        if not in_hull[c_index]:
            in_hull[c_index] = (
                (
                    (coordinate[0] >= xmin_grid_cells)
                    & (coordinate[0] <= xmax_grid_cells)
                )
                & (
                    (coordinate[1] >= ymin_grid_cells)
                    & (coordinate[1] <= ymax_grid_cells)
                )
                & (
                    (coordinate[2] >= zmin_grid_cells)
                    & (coordinate[2] <= zmax_grid_cells)
                )
            ).any()

    return in_hull
