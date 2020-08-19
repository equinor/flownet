from typing import Optional

import numpy as np


# pylint: disable=too-many-arguments,too-many-locals
def moller_trumbore(
    xstart: float,
    ystart: float,
    zstart: float,
    xend: float,
    yend: float,
    zend: float,
    v11: float,
    v12: float,
    v13: float,
    v21: float,
    v22: float,
    v23: float,
    v31: float,
    v32: float,
    v33: float,
) -> Optional[float]:
    """
    The Möller–Trumbore ray-triangle intersection algorithm, named after its inventors Tomas Möller and Ben Trumbore,
    is a fast method for calculating the intersection of a ray and a triangle in three dimensions without
    needing precomputation of the plane equation of the plane containing the triangle.

    In FlowNet the Möller–Trumbore is used to detect if and where flow tubes ("rays") are passing through faults
    (a collection of triangles).

    Args:
        xstart: X-coordinate of the start point of the tube
        ystart: Y-coordinate of the start point of the tube
        zstart: Z-coordinate of the start point of the tube
        xend: X-coordinate of the end point of the tube
        yend: Y-coordinate of the end point of the tube
        zend: Z-coordinate of the end point of the tube
        v11: Triangle corner 1 X-coordinate
        v12: Triangle corner 1 Y-coordinate
        v13: Triangle corner 1 Z-coordinate
        v21: Triangle corner 2 X-coordinate
        v22: Triangle corner 2 Y-coordinate
        v23: Triangle corner 2 Z-coordinate
        v31: Triangle corner 3 X-coordinate
        v32: Triangle corner 3 Y-coordinate
        v33: Triangle corner 3 Z-coordinate

    Returns:
        Either a float representing the fraction of the tube where the intersection happened or None

    """
    eps = 0.000001
    ray_start = np.array([xstart, ystart, zstart])
    ray_direction = np.array([xend, yend, zend]) - ray_start
    triangle = np.array([[v11, v12, v13], [v21, v22, v23], [v31, v32, v33]])

    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    pvec = np.cross(ray_direction, edge2)
    det = edge1.dot(pvec)

    if abs(det) < eps:
        return None

    inv_det = 1.0 / det
    t_vector = ray_start - triangle[0]
    u_value = t_vector.dot(pvec) * inv_det

    if u_value < 0.0 or u_value > 1.0:
        return None

    q_vector = np.cross(t_vector, edge1)
    v_value = ray_direction.dot(q_vector) * inv_det

    if v_value < 0.0 or u_value + v_value > 1.0:
        return None

    ray_fraction = edge2.dot(q_vector) * inv_det

    if ray_fraction < eps:
        return None

    if ray_fraction > 1:
        return None

    return ray_fraction
