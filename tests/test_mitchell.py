from typing import List

import numpy as np
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

from flownet.network_model._mitchell import (
    mitchell_best_candidate_modified_3d,
    scale_concave_hull_perforations,
)
from flownet.network_model._hull import check_in_hull
from src.flownet.utils.types import Coordinate


def test_mitchells_3d() -> None:
    well_perforations_3d = [
        [36.0, 452.0, 4014.0],
        [236.0, 420.0, 4014.0],
        [12.0, 276.0, 4014.0],
        [212.0, 228.0, 4014.0],
        [396.0, 276.0, 4014.0],
        [60.0, 68.0, 4014.0],
        [252.0, 12.0, 4014.0],
        [452.0, 44.0, 4014.0],
        [124.0, 340.0, 4002.0],
        [276.0, 316.0, 4002.0],
        [180.0, 124.0, 4002.0],
        [340.0, 140.0, 4002.0],
    ]
    num_added_flow_nodes = 10

    coordinates: List[Coordinate] = [
        tuple(elem)
        for elem in mitchell_best_candidate_modified_3d(
            well_perforations_3d,
            num_added_flow_nodes=num_added_flow_nodes,
            num_candidates=100,
            hull_factor=1.0,
            concave_hull_bounding_boxes=None,
            random_seed=999,
        )
    ]

    x_wells = [x[0] for x in well_perforations_3d]
    y_wells = [y[1] for y in well_perforations_3d]
    z_wells = [z[2] for z in well_perforations_3d]

    assert len(coordinates) == (len(well_perforations_3d) + num_added_flow_nodes)
    assert np.all([x[0] >= min(x_wells) for x in coordinates])
    assert np.all([x[0] <= max(x_wells) for x in coordinates])
    assert np.all([y[1] >= min(y_wells) for y in coordinates])
    assert np.all([y[1] <= max(y_wells) for y in coordinates])
    assert np.all([z[2] >= min(z_wells) for z in coordinates])
    assert np.all([z[2] <= max(z_wells) for z in coordinates])


def test_mitchells_2d() -> None:
    well_perforations_2d = [
        [36.0, 452.0, 4002.0],
        [236.0, 420.0, 4002.0],
        [12.0, 276.0, 4002.0],
        [212.0, 228.0, 4002.0],
        [396.0, 276.0, 4002.0],
        [60.0, 68.0, 4002.0],
        [252.0, 12.0, 4002.0],
        [452.0, 44.0, 4002.0],
        [124.0, 340.0, 4002.0],
        [276.0, 316.0, 4002.0],
        [180.0, 124.0, 4002.0],
        [340.0, 140.0, 4002.0],
    ]
    num_added_flow_nodes = 10

    coordinates: List[Coordinate] = [
        tuple(elem)
        for elem in mitchell_best_candidate_modified_3d(
            well_perforations_2d,
            num_added_flow_nodes=num_added_flow_nodes,
            num_candidates=100,
            hull_factor=1.0,
            concave_hull_bounding_boxes=None,
            random_seed=999,
        )
    ]

    x_wells = [x[0] for x in well_perforations_2d]
    y_wells = [y[1] for y in well_perforations_2d]
    z_wells = [z[2] for z in well_perforations_2d]

    assert len(coordinates) == (len(well_perforations_2d) + num_added_flow_nodes)
    assert np.all([x[0] >= min(x_wells) for x in coordinates])
    assert np.all([x[0] <= max(x_wells) for x in coordinates])
    assert np.all([y[1] >= min(y_wells) for y in coordinates])
    assert np.all([y[1] <= max(y_wells) for y in coordinates])
    assert np.all([z[2] == z_wells[0] for z in coordinates])


def test_hull_factor_mitchell() -> None:
    well_perforations_2d = [
        [36.0, 452.0, 4002.0],
        [236.0, 420.0, 4002.0],
        [12.0, 276.0, 4002.0],
        [212.0, 228.0, 4002.0],
        [396.0, 276.0, 4002.0],
        [60.0, 68.0, 4002.0],
        [252.0, 12.0, 4002.0],
        [452.0, 44.0, 4002.0],
        [124.0, 340.0, 4002.0],
        [276.0, 316.0, 4002.0],
        [180.0, 124.0, 4002.0],
        [340.0, 140.0, 4002.0],
    ]

    num_added_flow_nodes = 10
    hull_factor = 2.0

    coordinates: List[Coordinate] = [
        tuple(elem)
        for elem in mitchell_best_candidate_modified_3d(
            well_perforations_2d,
            num_added_flow_nodes=num_added_flow_nodes,
            num_candidates=100,
            hull_factor=hull_factor,
            concave_hull_bounding_boxes=None,
            random_seed=999,
        )
    ]

    x_hull, y_hull, z_hull = scale_concave_hull_perforations(
        well_perforations_2d, hull_factor
    )

    assert len(coordinates) == (len(well_perforations_2d) + num_added_flow_nodes)
    assert np.all([x[0] >= min(x_hull) for x in coordinates])
    assert np.all([x[0] <= max(x_hull) for x in coordinates])
    assert np.all([y[1] >= min(y_hull) for y in coordinates])
    assert np.all([y[1] <= max(y_hull) for y in coordinates])
    assert np.all([z[2] == z_hull[0] for z in coordinates])


def test_nodes_in_reservoir_volume_mitchells() -> None:
    well_perforations_3d = [
        [0.5, 0.5, 1.0],
        [0.5, 2.5, 0.5],
        [1.5, 5.5, 1.5],
        [3.5, 0.5, 1.0],
    ]

    concave_hull_bounding_boxes = np.array(
        [
            [0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
            [2.0, 3.0, 0.0, 2.0, 0.0, 2.0],
            [3.0, 5.0, 0.0, 2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0, 4.0, 0.0, 2.0],
            [0.0, 2.0, 4.0, 6.0, 0.0, 2.0],
            [2.0, 5.0, 4.0, 6.0, 0.0, 2.0],
            [5.0, 7.0, 4.0, 6.0, 0.0, 2.0],
            [5.0, 20.0, 6.0, 20.0, 0.0, 2.0],
        ]
    )

    coordinates: List[Coordinate] = [
        tuple(elem)
        for elem in mitchell_best_candidate_modified_3d(
            well_perforations_3d,
            num_added_flow_nodes=20,
            num_candidates=500,
            place_nodes_in_volume_reservoir=True,
            hull_factor=1.0,
            concave_hull_bounding_boxes=concave_hull_bounding_boxes,
            random_seed=999,
        )
    ]

    perforation_hull = Delaunay(np.array(well_perforations_3d))
    in_hull_perforations = perforation_hull.find_simplex(np.array(coordinates))

    in_hull_volume = check_in_hull(concave_hull_bounding_boxes, np.array(coordinates))
    assert in_hull_volume.all()
    assert any(x == -1 for x in in_hull_perforations)
