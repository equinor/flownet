from typing import List

import numpy as np
import pandas as pd

from ecl.grid import EclGrid

from ._tree import Tree
from ._create_egrid import create_egrid
from ._create_df_grid import create_df_grid


class CoarseGrid:
    def __init__(
        self, ecl_grid: EclGrid, well_coords: np.ndarray, partition: List[int]
    ):
        """
        Creates a 3D coarse grid

        FIXME:
        - Add types
        - Add decorators
        - Check lint
        - ZCORN assumes all cells in a layer have same z
        - Make sure tree splitting does not cut through wells in _find_split_plane and refine
        - Read partition from input
        """
        self._ecl_grid = ecl_grid
        self._well_coords: np.ndarray = well_coords
        self._partition: List[int] = partition

        # The grid
        self._coordinates: np.array = None
        self._actnum: np.array = None
        self._create_coarse_grid()

    @property
    def actnum(self) -> List[int]:
        return list(self._actnum)

    @property
    def coordinates(self) -> np.array:
        return self._coordinates

    @property
    def num_nodes(self) -> List[int]:
        return [len(x) for x in self.coordinates]

    @property
    def num_elements(self) -> List[int]:
        return [n - 1 for n in self.num_nodes]

    def ecl_grid(self) -> EclGrid:
        return create_egrid(
            self.coordinates, self.num_nodes, self.num_elements, self.actnum
        )

    def df_grid(self) -> pd.DataFrame:
        return create_df_grid(
            self.coordinates, self.num_nodes, self.num_elements, self.actnum
        )

    def _create_coarse_grid(self) -> None:
        """

        Create a coarse grid of the fine scale grid

        """

        # 1. Create tree with a root approximationg the bounding box
        # of the reservoir
        bboxmin, bboxmax = self.create_bbox()
        tree = Tree(bboxmin, bboxmax)

        # 2. Split the tree to obtain one well per node
        self.split(tree)

        # 3. Refine if prescribed partition is finer
        self.refine(tree)

        # 4. Smooth
        self.smooth(tree)

        # 5. Update actnum
        self.create_actnum()

    def split(self, tree) -> None:
        """
        Split the tree to obtain one well per node, suggesting to
        alternate directions in each split. If this doesn't work,
        another orientation of the split plane is used.

        """
        dim = 0
        plane, dim = self._find_split_plane(tree, dim)
        while plane is not None:
            tree.split(plane, dim)
            dim = np.remainder(dim + 1, 3)
            plane, dim = self._find_split_plane(tree, dim)

    def create_bbox(self) -> (List, List):
        """
        Create a bounding using vertices of the reservoir grid from the
        active cells
        """
        bboxmin = [np.finfo(float).max] * 3
        bboxmax = [np.finfo(float).min] * 3

        for cell in self._ecl_grid.cells(active=True):
            for i in range(8):
                x = self._ecl_grid.get_cell_corner(i, active_index=cell.active_index)
                for d in range(3):
                    bboxmin[d] = x[d] if x[d] < bboxmin[d] else bboxmin[d]
                    bboxmax[d] = x[d] if x[d] > bboxmax[d] else bboxmax[d]

        return bboxmin, bboxmax

    @staticmethod
    def extract_grid_data(tree) -> (List[np.array], np.array, np.array):
        """
        The grid is described by a list of points in x, y and z
        direction (the coordinates list). For convenience this
        function also returns the number of nodes and the number of
        elements
        """
        leaves = tree.get_leaves()
        num_leaves = len(leaves)

        # Find the minimum coordinate of each leaf
        xmin = np.ndarray((num_leaves, 3))
        for k, node in enumerate(leaves):
            xmin[k, :] = node.xmin

        num_nodes = np.empty(3, "int64")
        coordinates = [None] * 3

        # # Possibly filter points that are close. If we do this, then
        # # we should make sure there is still only one well per node
        # tol_same_node = 1e-9
        # decimals = int(round(abs(np.log10(max(tree.root.dx())*tol_same_node))))
        # for d in range(3):
        #     coordinates[d] = np.unique(xmin[:,d].round(decimals=decimals))
        #     coordinates[d] = np.append(coordinates[d], tree.root.xmax[d])
        #     num_nodes[d] = len(coordinates[d])

        for dim in range(3):
            coordinates[dim] = np.unique(xmin[:, dim])
            coordinates[dim] = np.append(coordinates[dim], tree.root.xmax[dim])
            num_nodes[dim] = len(coordinates[dim])

        assert np.prod(num_nodes - 1) == num_leaves

        # Elements
        num_elements = num_nodes - 1

        # Return nodal coordinates dimension-wise, number of nodes in
        # each dimension, and number of elements
        return coordinates, num_nodes, num_elements

    def _find_split_plane(self, tree, dim) -> (List, int):
        """
        Find a plane needed for splitting the tree. The input dim is the
        suggested plane orientation.
        """
        checked_nodes = []
        checked_wc = []

        # Tolerance for determining two wells as close to each other
        # in a single dimension
        # FIXME how to choose this tolerance
        tol = 0.1

        for well_coord in self._well_coords:
            node = tree.locate(well_coord)

            assert node is not None, (
                "Well with coordinate {well_coord} is not found in the tree with size"
                "{tree.root.xmin, tree.root.xmax}. This should not happen."
            )

            try:
                idx = checked_nodes.index(node)
            except ValueError:
                idx = None

            if idx is not None:
                # Node has already been found and is needed to be
                # split. We have several options which plane to
                # choose: split by the cell centroid or use something
                # smarter using the well locations.

                # Split cell centroid: this will create a lot of nodes in
                # the tree and no guarantees for not splitting exactly at
                # the wells
                # return node.midpoint(), dim

                # Split at dim unless two well points are too close (NB
                # there is no guarantee for the other wells)
                plane = node.midpoint()
                wc0 = checked_wc[idx]
                w_dist = abs(wc0 - well_coord)
                w_mid = 0.5 * (wc0 + well_coord)
                if w_dist[dim] < tol:
                    # The two well coordinates are close along
                    # dimension dim. Don't split in along this
                    # dimension, but suggest the dimension with
                    # maximum distance.
                    split_dim = np.argmax(w_dist)
                    plane[split_dim] = w_mid[split_dim]
                    return plane, split_dim

                plane[dim] = w_mid[dim]
                return plane, dim

                # FIXME: Split at the dimension where the two well
                # points are farthest from each other.

            checked_nodes.append(node)
            checked_wc.append(well_coord)

        # No node is needed to be split
        plane = None
        dim = None
        return plane, dim

    def refine(self, tree) -> None:
        """
        Refine the tree if the desired partition is finer than the
        current partition of the tree
        """
        coordinates, _, num_elements = self.extract_grid_data(tree)

        for dim in range(3):
            for _ in range(self._partition[dim] - num_elements[dim]):
                # Split the longest section
                # FIXME split smarter wrt well coordinates
                diff = np.diff(coordinates[dim])
                idx = diff.argmax()
                xcut = tree.root.midpoint()
                xcut[dim] = coordinates[dim][idx] + 0.5 * diff[idx]
                tree.split(xcut, dim, tree.root)
                coordinates, _, num_elements = self.extract_grid_data(tree)

    @staticmethod
    def _laplacian_smoothing(
        x, xfix, A=None, rfix=None
    ) -> (np.array, np.ndarray, np.ndarray, float):
        """
        Perform a few laplacian smoothing steps as y=A*x and rfix
        are the indices of the fixed points
        """
        initial_x = np.concatenate([xfix, x])
        idx = np.argsort(initial_x)
        initial_x = initial_x[idx]

        if A is None or rfix is None:
            nfix = np.arange(len(xfix), dtype=int)
            n = len(initial_x)
            rfix = np.in1d(idx, nfix)

            A = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
            A[rfix, :] = 0
            A += 2 * np.diag(rfix)
            A *= 0.5

        y = A.dot(initial_x)
        err = np.linalg.norm(initial_x - y)
        return y[np.invert(rfix)], A, rfix, err

    def smooth(self, tree) -> Tree:
        """
        Smooth one dimension at a time using a laplacian smoother (averaging)
        """
        coordinates, _, _ = self.extract_grid_data(tree)
        new_cuts = [None] * 3

        # Only smooth until l2 norm is smaller than tol or until maxit
        # is reached
        tol = 1e-2
        maxit = 100

        for dim in range(3):
            # Remove boundary from the smoothing (these are the root
            # node coordinates)
            x = coordinates[dim][1:-1]

            # Add the removed root node coordinates to the list of
            # fixed points
            xfix = np.array(coordinates[dim][0])
            xfix = np.append(xfix, self._well_coords[:, dim])
            xfix = np.append(xfix, coordinates[dim][-1])

            # Do one round of smoothing to set up the matrix
            y, A, rfix, err = self._laplacian_smoothing(x, xfix, None, None)
            x = y.copy()
            it = 1

            while err / max(tree.root.dx()) > tol and it < maxit:
                y, _, _, err = self._laplacian_smoothing(x, xfix, A, rfix)
                x = y.copy()
                it += 1
            new_cuts[dim] = x

        # Construct a new tree
        tree2 = Tree(tree.root.xmin, tree.root.xmax)
        for dim in range(3):
            midpoint = tree.root.midpoint()
            for cut in new_cuts[dim]:
                midpoint[dim] = cut
                tree2.split(midpoint, dim)

        # Update self. It's sufficient to update coordinates since the
        # topology is not changed
        self._coordinates, _, _ = self.extract_grid_data(tree2)

        return tree2

    def _in_ecl_grid(self, x, start_ijk=None) -> (bool, tuple):
        """
        See if x is in the grid by calling a relevant ecl function.
        """
        ijk = self._ecl_grid.find_cell(x[0], x[1], x[2], start_ijk=start_ijk)

        if ijk is None:
            return False, None

        if self._ecl_grid.active(ijk):
            return True, ijk

        return False, None

    def create_actnum(self) -> None:
        """
        Check centroid to determine if cell is inside/outside (we could
        also check the corners).

        Note how we loop over the elements dimension-wise, as this
        defines the element numbering

        """
        self._actnum = [0] * np.prod(self.num_elements)
        num_nodes = self.num_nodes
        coords = self.coordinates
        start_ijk = None
        xyz = [0] * 3
        cnt = 0

        for k in range(num_nodes[2] - 1):
            xyz[2] = 0.5 * (coords[2][k] + coords[2][k + 1])
            for j in range(num_nodes[1] - 1):
                xyz[1] = 0.5 * (coords[1][j] + coords[1][j + 1])
                for i in range(num_nodes[0] - 1):
                    xyz[0] = 0.5 * (coords[0][i] + coords[0][i + 1])
                    inside, start_ijk = self._in_ecl_grid(xyz, start_ijk)
                    if inside:
                        self._actnum[cnt] = 1
                        cnt += 1
