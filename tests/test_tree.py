from typing import List
from itertools import permutations

from flownet.coarse_model._tree import Tree, Node


def check_leaves(leaves, corners):
    for node in leaves:
        x0 = node.xmin
        x1 = node.xmax
        corner = [x0, x1]
        for c in corners:
            if corner == c:
                return True
        return False


def single_tree() -> (Tree, List):
    tree = Tree([0, 0, 0], [10, 10, 10])
    xplane = [5, 0, 0]
    corners = [[[0, 0, 0], [5, 10, 10]], [[5, 0, 0], [10, 10, 10]]]
    tree.split(xplane, 0)
    return tree, corners


def get_permutated_trees() -> List[Tree]:
    xmin = [0, 0, 0]
    xmax = [10, 10, 10]
    xplane = [5, 0, 0]
    yplane = [0, 3, 0]
    zplane = [0, 0, 4]
    planes = [(xplane, 0), (yplane, 1), (zplane, 2)]
    all_planes = list(permutations(planes))
    trees = [None] * len(all_planes)

    # We should get these 8 cells
    corners = [
        [[0, 0, 0], [5, 3, 4]],
        [[5, 0, 0], [10, 3, 4]],
        [[0, 3, 0], [5, 10, 4]],
        [[5, 3, 0], [10, 10, 4]],
        [[0, 0, 4], [5, 3, 10]],
        [[5, 0, 4], [10, 3, 10]],
        [[0, 3, 4], [5, 10, 10]],
        [[5, 3, 4], [10, 10, 10]],
    ]

    for k, plane in enumerate(all_planes):
        xplane, xdim = plane[0]
        yplane, ydim = plane[1]
        zplane, zdim = plane[2]
        tree = Tree([0, 0, 0], [10, 10, 10])
        tree.split(xplane, xdim)
        tree.split(yplane, ydim)
        tree.split(zplane, zdim)
        trees[k] = tree

    return trees, corners


def test_x() -> None:
    tree, corners = single_tree()
    leaves = tree.get_leaves()
    assert check_leaves(leaves, corners)
    assert tree.volume() == 1000
    assert tree.volume(leaves) == 1000
    assert tree.area() == 800
    assert tree.area(leaves) == 800


def test_xz() -> None:
    xplane = [5, 0, 0]
    zplane = [0, 0, 4]

    # We should get these four cells
    corners = [
        [[0, 0, 0], [5, 10, 4]],
        [[5, 0, 0], [10, 10, 4]],
        [[0, 0, 4], [5, 10, 10]],
        [[5, 0, 4], [10, 10, 10]],
    ]

    # Split in x then z
    tree = Tree([0, 0, 0], [10, 10, 10])
    tree.split(xplane, 0)
    tree.split(zplane, 2)
    assert check_leaves(tree.get_leaves(), corners)
    assert tree.volume() == 1000
    assert tree.area() == 1000


def test_zx() -> None:
    xplane = [5, 0, 0]
    zplane = [0, 0, 4]

    # We should get these four cells
    corners = [
        [[0, 0, 0], [5, 10, 4]],
        [[5, 0, 0], [10, 10, 4]],
        [[0, 0, 4], [5, 10, 10]],
        [[5, 0, 4], [10, 10, 10]],
    ]

    # Split in z then x
    tree = Tree([0, 0, 0], [10, 10, 10])
    tree.split(zplane, 2)
    tree.split(xplane, 0)
    assert check_leaves(tree.get_leaves(), corners)
    assert tree.area() == 1000
    return tree


def test_xyz() -> None:
    trees, corners = get_permutated_trees()
    for tree in trees:
        assert check_leaves(tree.get_leaves(), corners)
        assert tree.volume() == 1000
        assert tree.area() == 1200


def test_locate1():
    trees, _ = get_permutated_trees()
    for tree in trees:
        node = tree.locate([0, 0, -1])
        assert node is None
        x = [9, 9, 9]
        node = tree.locate(x)
        assert node is not None
        assert node.is_leaf()
        assert node.inside(x)


def test_locate2():
    trees, _ = get_permutated_trees()
    for tree in trees:
        tree.split([2, 0, 0], 0)
        tree.split([1, 0, 0], 0)
        x = [0.1, 0.1, 0.1]
        node = tree.locate(x)
        assert node is not None
        assert node.is_leaf()
        assert node.inside(x)
