import collections

from pytest import approx
import numpy as np
import pandas as pd

from flownet.network_model._generate_connections import (
    _generate_connections,
    _create_entity_connection_matrix,
    _is_angle_too_large,
    _split_additional_flow_nodes,
)

DATA = {
    "WELL_NAME": ["P1", "P2", "P3", "P4", "P5"],
    "X": [1, 1, 3, 2, 1],
    "Y": [1, 2, 3, 4, 5],
    "Z": [2, 1, 3, 3, 2],
}
DF_COORDINATES = pd.DataFrame(DATA)

STARTS_NONE = [
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (1.0, 2.0, 1.0),
    (1.0, 2.0, 1.0),
    (1.0, 2.0, 1.0),
    (3.0, 3.0, 3.0),
    (3.0, 3.0, 3.0),
    (3.0, 3.0, 3.0),
    (3.0, 3.0, 3.0),
    (2.0, 4.0, 3.0),
    (2.0, 4.0, 3.0),
    (2.0, 4.0, 3.0),
    (1.0, 5.0, 2.0),
    (1.0, 5.0, 2.0),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
]
ENDS_NONE = [
    (1.0, 2.0, 1.0),
    (3.0, 3.0, 3.0),
    (2.0, 4.0, 3.0),
    (1.0, 5.0, 2.0),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
    (3.0, 3.0, 3.0),
    (1.0, 5.0, 2.0),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (2.0, 4.0, 3.0),
    (1.0, 5.0, 2.0),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
    (1.0, 5.0, 2.0),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
]
STARTS_30 = [
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (1.0, 2.0, 1.0),
    (1.0, 2.0, 1.0),
    (3.0, 3.0, 3.0),
    (3.0, 3.0, 3.0),
    (3.0, 3.0, 3.0),
    (3.0, 3.0, 3.0),
    (2.0, 4.0, 3.0),
    (2.0, 4.0, 3.0),
    (2.0, 4.0, 3.0),
    (1.0, 5.0, 2.0),
    (1.0, 5.0, 2.0),
]
ENDS_30 = [
    (3.0, 3.0, 3.0),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
    (3.0, 3.0, 3.0),
    (1.0, 5.0, 2.0),
    (2.0, 4.0, 3.0),
    (1.0, 5.0, 2.0),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
    (1.0, 5.0, 2.0),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
    (1.3725204227553418, 2.5870698969226797, 1.8383890288065896),
    (1.5398557835300521, 2.7123647594851796, 2.326882995636896),
]


def test_generate_connections() -> None:

    config = collections.namedtuple("configuration", "flownet")
    config.flownet = collections.namedtuple("flownet", "additional_flow_nodes")
    config.flownet.additional_flow_nodes = 2
    config.flownet.additional_node_candidates = 2
    config.flownet.place_nodes_in_volume_reservoir = None
    config.flownet.hull_factor = 1
    config.flownet.random_seed = 1
    config.flownet.mitchells_algorithm = "normal"
    config.flownet.angle_threshold = None

    # pylint: disable=no-member
    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES,
        configuration=config,
        additional_flow_nodes=config.flownet.additional_flow_nodes,
    )

    assert len(starts) == len(ends)
    assert all(starts[i] == approx(STARTS_NONE[i]) for i in range(len(starts)))
    assert all(ends[i] == approx(ENDS_NONE[i]) for i in range(len(ends)))

    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES,
        configuration=config,
        additional_flow_nodes=config.flownet.additional_flow_nodes,
        concave_hull_bounding_boxes=np.array([0, 2, 0, 2, 0, 2]).reshape(-1, 6),
    )

    assert len(starts) == len(ends)
    assert len(starts) != len(STARTS_NONE)
    assert len(ends) != len(ENDS_NONE)

    # Test removal of some connections
    config.flownet.angle_threshold = 150

    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES,
        configuration=config,
        additional_flow_nodes=config.flownet.additional_flow_nodes,
    )

    assert len(starts) == len(ends)
    assert len(starts) != len(STARTS_NONE)
    assert len(ends) != len(ENDS_NONE)

    # Test removal of all connections
    config.flownet.angle_threshold = 1
    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES,
        configuration=config,
        additional_flow_nodes=config.flownet.additional_flow_nodes,
    )

    assert len(starts) == len(ends) == 0


def test_split_additional_flow_nodes() -> None:

    total_additional_nodes = 100
    concave_hull_list = [
        np.array([[0, 2, 0, 2, 0, 2], [2, 4, 0, 2, 0, 2]]),
        np.array([0, 2, 0, 2, 2, 4]),
    ]

    sep_add_flownodes = _split_additional_flow_nodes(
        total_additional_nodes=total_additional_nodes,
        concave_hull_list=concave_hull_list,
    )

    assert sum(sep_add_flownodes) == total_additional_nodes
    assert sep_add_flownodes[0] == 67
    assert sep_add_flownodes[1] == 33


def test_create_entity_connection_matrix() -> None:

    df = _create_entity_connection_matrix(
        DF_COORDINATES,
        STARTS_30,
        ENDS_30,
        [],
        [],
        max_distance=10,
        max_distance_fraction=0,
    )
    assert len(df) == 13
    for well in DATA["WELL_NAME"]:
        assert df["start_entity"].str.contains(well).any()

    df = _create_entity_connection_matrix(
        DF_COORDINATES,
        STARTS_30,
        ENDS_30,
        [],
        [],
        max_distance=2.9,
        max_distance_fraction=0,
    )
    assert len(df) == 10

    df = df = _create_entity_connection_matrix(
        DF_COORDINATES,
        STARTS_30,
        ENDS_30,
        [],
        [],
        max_distance=2.9,
        max_distance_fraction=0.9,
    )
    assert len(df) == 2

    df = df = _create_entity_connection_matrix(
        DF_COORDINATES,
        STARTS_30,
        ENDS_30,
        [],
        [],
        max_distance=2.9,
        max_distance_fraction=1,
    )
    assert len(df) == 0


def test_is_angle_too_large() -> None:
    assert _is_angle_too_large(0, 1, 1, 2 ** 0.5)
    assert not _is_angle_too_large(180, 1, 1, 2 ** 0.5)
    assert _is_angle_too_large(44, 2 ** 0.5, 1, 1)
    assert not _is_angle_too_large(45, 2 ** 0.5, 1, 1)
    assert not _is_angle_too_large(46, 2 ** 0.5, 1, 1)
