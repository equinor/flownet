from typing import List

import numpy as np
from flownet.network_model._mitchell import mitchell_best_candidate_modified_3d
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
            hull_factor=1.2,
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
            num_added_flow_nodes=10,
            num_candidates=100,
            hull_factor=1.2,
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
