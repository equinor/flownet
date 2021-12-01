from typing import List, Tuple
import numpy as np
import pytest

from flownet.coarse_model import CoarseGrid, create_egrid, Tree
from ecl.grid import EclGrid


def get_coarse_grid(partition: List) -> Tuple:
    data_dir = "./tests/data/"

    grid_file = data_dir + "faarikaal2.EGRID"
    well_file = data_dir + "faarikaal2.txt"

    ecl_grid = EclGrid(grid_file)
    well_coords = np.loadtxt(well_file)

    # Make sure well_coords is sufficiently large
    if len(well_coords.shape) == 1:
        well_coords = np.reshape(well_coords, (1, well_coords.shape[0]))

    CG = CoarseGrid(ecl_grid, well_coords, partition)
    return CG, well_coords


def check_bbox(coordinates, bboxmin, bboxmax) -> None:
    tol = 1e-3
    for d in range(3):
        assert abs(coordinates[d][0] - bboxmin[d]) < tol
        assert abs(coordinates[d][-1] - bboxmax[d]) < tol


def check_tree(num_elements, tree) -> None:
    assert np.prod(num_elements) == len(tree.get_leaves())


def test_coarse_grid() -> None:
    # Check that we have reasonable values at the various stages
    CG, _ = get_coarse_grid([1, 1, 1])

    # 1.
    bboxmin, bboxmax = CG.create_bbox()
    tree = Tree(bboxmin, bboxmax)

    # 2.
    CG.split(tree)

    # 3.
    CG.refine(tree)

    # 4.
    tree = CG.smooth(tree)

    # 5.
    CG.create_actnum()

    # 6.
    coordinates, num_nodes, num_elements = CG.extract_grid_data(tree)

    # Checks
    check_bbox(coordinates, bboxmin, bboxmax)
    check_tree(num_elements, tree)


def check_one_well_per_cell(cg, well_coords) -> bool:
    num_elements = cg.nx * cg.ny * cg.nz
    cnt = np.zeros(num_elements, dtype=int)
    for well_coord in well_coords:
        ijk = cg.find_cell(well_coord[0], well_coord[1], well_coord[2])
        assert ijk is not None
        cnt[cg.get_global_index(ijk)] += 1
    return max(cnt) == 1


def test_one_well_per_cell() -> None:
    # Test that there are only one well per cell in the coarse grid
    CG, well_coords = get_coarse_grid([1, 1, 1])
    cg = CG.ecl_grid()
    assert check_one_well_per_cell(cg, well_coords)


@pytest.mark.xfail
def test_well_setup() -> None:
    # Test of a particular well setup that can be problematic
    dims = (1, 1, 1)
    dV = (1, 1, 1)
    ecl_grid = EclGrid.create_rectangular(dims, dV)
    well_coords = np.array([[0.25, 0.25, 0.5], [0.25, 0.75, 0.5], [0.75, 0.5, 0.5]])
    CG = CoarseGrid(ecl_grid, well_coords, [1, 1, 1])
    cg = CG.ecl_grid()
    assert check_one_well_per_cell(cg, well_coords)
