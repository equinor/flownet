import pytest

from flownet.network_model._mitchell import mitchell_best_candidate_modified_3d
from src.flownet.utils.types import Coordinate


def test_mitchells_3D() -> None:
    well_perforations_3D =[[36.0, 452.0, 4014.0], [236.0, 420.0, 4014.0], [12.0, 276.0, 4014.0], [212.0, 228.0, 4014.0], [396.0, 276.0, 4014.0], [60.0, 68.0, 4014.0], [252.0, 12.0, 4014.0], [452.0, 44.0, 4014.0], [124.0, 340.0, 4002.0], [276.0, 316.0, 4002.0], [180.0, 124.0, 4002.0], [340.0, 140.0, 4002.0]]
    coordinates: List[Coordinate] = [
        tuple(elem)
        for elem in mitchell_best_candidate_modified_3d(
            well_perforations_3D,
            num_added_flow_nodes=10,
            num_candidates=100,
            hull_factor=1.2,
            concave_hull_bounding_boxes=None,
            random_seed=999,
        )
    ]


def test_mitchells_2D() -> None:
    well_perforations_2D = [[36.0, 452.0, 4002.0], [236.0, 420.0, 4002.0], [12.0, 276.0, 4002.0], [212.0, 228.0, 4002.0], [396.0, 276.0, 4002.0], [60.0, 68.0, 4002.0], [252.0, 12.0, 4002.0], [452.0, 44.0, 4002.0], [124.0, 340.0, 4002.0], [276.0, 316.0, 4002.0], [180.0, 124.0, 4002.0], [340.0, 140.0, 4002.0]]
        coordinates: List[Coordinate] = [
        tuple(elem)
        for elem in mitchell_best_candidate_modified_3d(
            well_perforations_3D,
            num_added_flow_nodes=10,
            num_candidates=100,
            hull_factor=1.2,
            concave_hull_bounding_boxes=None,
            random_seed=999,
        )
    ]



