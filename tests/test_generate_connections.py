import collections

from pytest import approx
import numpy as np
import pandas as pd

from flownet.network_model._generate_connections import (
    _generate_connections,
    _create_entity_connection_matrix,
    _is_angle_too_large,
    _create_record,
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
    config.flownet.hull_factor = 1
    config.flownet.random_seed = 1
    config.flownet.angle_threshold = None

    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES, configuration=config
    )

    assert len(starts) == len(ends)
    assert all([starts[i] == approx(STARTS_NONE[i]) for i in range(len(starts))])
    assert all([ends[i] == approx(ENDS_NONE[i]) for i in range(len(ends))])

    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES,
        configuration=config,
        concave_hull_bounding_boxes=np.array([0, 2, 0, 2, 0, 2]).reshape(-1, 6),
    )

    assert len(starts) == len(ends)
    assert len(starts) != len(STARTS_NONE)
    assert len(ends) != len(ENDS_NONE)

    # Test removal of some connections
    config.flownet.angle_threshold = 30

    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES, configuration=config
    )

    assert len(starts) == len(ends)
    assert all([starts[i] == approx(STARTS_30[i]) for i in range(len(starts))])
    assert all([ends[i] == approx(ENDS_30[i]) for i in range(len(ends))])

    # Test removal of all connections
    config.flownet.angle_threshold = 180
    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES, configuration=config
    )

    assert len(starts) == len(ends) == 0


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
    assert not _is_angle_too_large(0, 1, 1, 2 ** 0.5)
    assert not _is_angle_too_large(90, 100, 1, (1 ** 2 + 100 ** 2) ** 0.5)
    assert _is_angle_too_large(91, 100, 1, (1 ** 2 + 100 ** 2) ** 0.5)
