import collections

import numpy as np
import pandas as pd

from flownet.network_model._generate_connections import (
    _generate_connections,
    _create_entity_connection_matrix,
)

DATA = {
    "WELL_NAME": ["P1", "P2", "P3", "P4", "P5"],
    "X": [1, 1, 3, 2, 1],
    "Y": [1, 2, 3, 4, 5],
    "Z": [2, 1, 3, 3, 2],
}
DF_COORDINATES = pd.DataFrame(DATA)

STARTS = [
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
ENDS = [
    (1.0, 2.0, 1.0),
    (3.0, 3.0, 3.0),
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


def test_generate_connections() -> None:

    config = collections.namedtuple("configuration", "flownet")
    config.flownet = collections.namedtuple("flownet", "additional_flow_nodes")  # type: ignore
    config.flownet.additional_flow_nodes = 2
    config.flownet.additional_node_candidates = 2
    config.flownet.hull_factor = 1
    config.flownet.random_seed = 1

    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES, configuration=config
    )

    assert len(starts) == len(ends)
    assert starts == STARTS
    assert ends == ENDS

    starts, ends = _generate_connections(
        df_coordinates=DF_COORDINATES,
        configuration=config,
        concave_hull_bounding_boxes=np.array([0, 2, 0, 2, 0, 2]).reshape(-1, 6),
    )

    assert len(starts) == len(ends)
    assert starts is not STARTS
    assert ends is not ENDS


def test_create_entity_connection_matrix() -> None:

    df = _create_entity_connection_matrix(
        DF_COORDINATES, STARTS, ENDS, [], [], max_distance=10, max_distance_fraction=0
    )
    assert len(df) == 16
    for well in DATA["WELL_NAME"]:  # type: ignore
        assert df["start_entity"].str.contains(well).any()

    df = _create_entity_connection_matrix(
        DF_COORDINATES, STARTS, ENDS, [], [], max_distance=2.9, max_distance_fraction=0
    )
    assert len(df) == 13

    df = df = _create_entity_connection_matrix(
        DF_COORDINATES,
        STARTS,
        ENDS,
        [],
        [],
        max_distance=2.9,
        max_distance_fraction=0.9,
    )
    assert len(df) == 2

    df = df = _create_entity_connection_matrix(
        DF_COORDINATES, STARTS, ENDS, [], [], max_distance=2.9, max_distance_fraction=1
    )
    assert len(df) == 0
