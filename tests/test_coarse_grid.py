from typing import List, Dict
import numpy as np
from flownet.coarse_model import CoarseGrid, create_egrid
from ecl.grid import EclGrid


def get_coarse_grid(partition: List) -> Dict:
    data_dir = "./tests/data/"

    grid_file = data_dir + "faarikaal2.EGRID"
    well_file = data_dir + "faarikaal2.txt"

    ecl_grid = EclGrid(grid_file)
    well_coords = np.loadtxt(well_file)

    # Make sure well_coords is sufficiently large
    if len(well_coords.shape) == 1:
        well_coords = np.reshape(well_coords, (1, well_coords.shape[0]))

    CG = CoarseGrid(ecl_grid, well_coords, partition)
    cg_dict = CG.create_coarse_grid()
    cg = create_egrid(
        cg_dict["coordinates"],
        cg_dict["num_nodes"],
        cg_dict["num_elements"],
        cg_dict["actnum"],
    )
    return cg, well_coords, cg_dict, CG


def test_egrid_construction() -> None:
    cg, _, _, _ = get_coarse_grid([1, 1, 1])
    num_elements = cg.nx * cg.ny * cg.nz
    num_nodes = (cg.nx + 1) * (cg.ny + 1) * (cg.nz + 1)
    assert cg["num_elements"] == num_elements
    assert cg["num_nodes"] == num_nodes


def test_coarse_grid_bbox() -> None:
    cg, _, _, CG = get_coarse_grid([1, 1, 1])
    bboxmin, bboxmax = CG.create_bbox()
    x = cg["coordinates"]
    tol = 1e-3
    for d in range(3):
        assert abs(x[d][0] - bboxmin[d]) < tol
        assert abs(x[d][-1] - bboxmax[d]) < tol


def test_coarse_grid_points() -> None:
    cg, well_coords, _, _ = get_coarse_grid([1, 1, 1])
    num_elements = cg.nx * cg.ny * cg.nz
    cnt = np.zeros(num_elements, dtype=int)
    for well_coord in well_coords:
        ijk = cg.find_cell(wc[0], wc[1], wc[2])
        assert ijk is not None
        cnt[cg.get_global_index(ijk)] += 1
    assert max(cnt) == 1
