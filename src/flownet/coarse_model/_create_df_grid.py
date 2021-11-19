import numpy as np
import pandas as pd


def create_df_grid(coordinates, num_nodes, num_elements, actnum=None) -> pd.DataFrame:
    """
    This function creates a pd.DataFrame grid matching the grid
    created in NetworkModel (NetworkModel._grid)

    The numbering of the elements must match the numbering of actnum,
    i.e., a loop over z first, then y and last x.

    The numbering of the nodes in each element must match the
    numbering in NetworkModel. FIXME

    """

    nel = np.prod(num_elements)
    element_coords = np.empty((nel, 24))
    xyz = np.empty((3))
    cnt = 0

    for k in range(num_nodes[2] - 1):
        for j in range(num_nodes[1] - 1):
            for i in range(num_nodes[0] - 1):
                # ijk is the lower-left corner
                for k_loc in range(2):
                    xyz[2] = coordinates[2][k + k_loc]
                    for j_loc in range(2):
                        xyz[1] = coordinates[1][j + j_loc]
                        for i_loc in range(2):
                            xyz[0] = coordinates[0][i + i_loc]
                            # xyz is the coordinate of the corner
                            idx = 3 * (i_loc + 2 * j_loc + 4 * k_loc)
                            element_coords[cnt, idx : (idx + 3)] = xyz
                cnt += 1

    header = [c + str(i) for i in range(8) for c in ["x", "y", "z"]]
    df = pd.DataFrame(columns=header, data=element_coords)

    if actnum is None:
        actnum = np.ones(nel, dtype=int)
    df["ACTNUM"] = actnum

    return df
