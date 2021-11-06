import numpy as np

from ecl.grid import EclGrid

from ..network_model._create_egrid import construct_kw


def create_egrid(coordinates, num_nodes, num_elements, actnum=None):
    """This function writes a grid based on the list of coordinates (in x,
    y and z).

    """

    gdim = 3
    assert len(coordinates) == gdim
    assert len(num_nodes) == gdim
    assert len(num_elements) == gdim

    # COORD
    coord = np.empty((num_nodes[0] * num_nodes[1], 2 * gdim))
    zmin = min(coordinates[2])
    zmax = max(coordinates[2])
    cnt = 0
    for j in range(num_nodes[1]):
        for i in range(num_nodes[0]):
            coord[cnt, 0:gdim] = np.array([coordinates[0][i], coordinates[1][j], zmin])
            coord[cnt, gdim : 2 * gdim] = np.array(
                [coordinates[0][i], coordinates[1][j], zmax]
            )
            cnt += 1

    # ZCORN
    # FIXME Here we assume all cells in a layer have same z
    # Elements
    n_xy = 2 * num_elements[0] * 2 * num_elements[1]
    zcorn = np.empty((2 * num_elements[2], n_xy))
    cnt = 0
    for k in range(num_elements[2]):
        zcorn[cnt, :] = coordinates[2][k] * np.ones(n_xy)
        zcorn[cnt + 1, :] = coordinates[2][k + 1] * np.ones(n_xy)
        cnt += 2

    # ACTNUM
    if actnum is None:
        actnum = np.ones(np.prod(num_elements), dtype=np.int32)

    return EclGrid.create(
        num_elements,
        construct_kw("ZCORN", zcorn.flatten().tolist()),
        construct_kw("COORD", coord.flatten().tolist()),
        construct_kw("ACTNUM", actnum.flatten().tolist(), int_type=True),
    )
