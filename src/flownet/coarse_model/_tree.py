from typing import List


class Node:
    def __init__(self, xmin: List[float], xmax: List[float]):
        self.xmin: List[float] = xmin.copy()
        self.xmax: List[float] = xmax.copy()
        self.left: Node = None
        self.right: Node = None

    def midpoint(self) -> List[float]:
        m = self.xmin.copy()
        for d in range(len(self.xmin)):
            m[d] += self.xmax[d]
            m[d] *= 0.5
        return m

    def dx(self) -> List[float]:
        m = self.xmax.copy()
        for d in range(len(self.xmin)):
            m[d] -= self.xmin[d]
        return m

    def split(self, coord: List[float], dim: int) -> None:
        # Sufficient to pass the coordinate instead of the whole coord
        # vector
        if self.xmin[dim] < coord[dim] and coord[dim] < self.xmax[dim]:
            leftmax = self.xmax.copy()
            leftmax[dim] = coord[dim]
            self.left = Node(self.xmin, leftmax)

            rightmin = self.xmin.copy()
            rightmin[dim] = coord[dim]
            self.right = Node(rightmin, self.xmax)
        else:
            pass

    def is_leaf(self) -> bool:
        if self.left is not None or self.right is not None:
            return False
        return True

    def inside(self, x: List[float]) -> bool:
        for dim in range(len(self.xmin)):
            if x[dim] < self.xmin[dim] or x[dim] > self.xmax[dim]:
                return None
        return self

    def __str__(self) -> str:
        if self is None:
            self_id = -1
        else:
            self_id = self.ptr()
        if self.left is None:
            left_id = -1
        else:
            left_id = self.left.ptr()
        if self.right is None:
            right_id = -1
        else:
            right_id = self.right.ptr()
        return (
            f"{self_id=}"
            + " "
            + str(self.xmin)
            + " "
            + str(self.xmax)
            + " "
            + f"{left_id=}"
            + " "
            + f"{right_id=}"
        )

    def ptr(self) -> str:
        return hex(id(self))

    def volume(self) -> float:
        vol = 1
        for d in range(len(self.xmin)):
            vol = vol * (self.xmax[d] - self.xmin[d])
        return vol

    def area(self) -> float:
        dx = self.xmax[0] - self.xmin[0]
        dy = self.xmax[1] - self.xmin[1]
        if len(self.xmax) == 3:
            dz = self.xmax[2] - self.xmin[2]
            return 2 * (dx * dy + dx * dz + dy * dz)
        return 2 * (dx + dy)


class Tree:
    def __init__(self, xmin: List[float], xmax: List[float]):
        self.root: Node = Node(xmin, xmax)
        self.gdim: int = len(self.root.xmin)

    def get(self, nodes: List[Node], node: Node = None) -> List[Node]:
        if node is None:
            self.get(nodes, self.root)
        else:
            nodes.append(node)
            if node.left is not None:
                self.get(nodes, node.left)
            if node.right is not None:
                self.get(nodes, node.right)
        return nodes

    def get_leaves(self, leaves=None, node=None) -> List[Node]:
        if leaves is None and node is None:
            leaves = []
            self.get_leaves(leaves, self.root)
        else:
            if node.is_leaf():
                leaves.append(node)
            else:
                if node.left is not None:
                    self.get_leaves(leaves, node.left)
                if node.right is not None:
                    self.get_leaves(leaves, node.right)
        return leaves

    def split(self, coord: List[float], dim: int, node: Node = None) -> None:
        if node is None:
            self.split(coord, dim, self.root)
        else:
            if node.is_leaf():
                node.split(coord, dim)
            else:
                if node.left is not None:
                    self.split(coord, dim, node.left)
                if node.right is not None:
                    self.split(coord, dim, node.right)

    def locate(self, x: List[float], node: Node = None) -> Node:
        if node is None:
            return self.locate(x, self.root)
        else:
            if node.is_leaf():
                return node.inside(x)

            if node.left is not None:
                left = self.locate(x, node.left)
            if node.right is not None:
                right = self.locate(x, node.right)

        for n in [left, right]:
            if n is not None and n.is_leaf() and n.inside(x):
                return n
        return None

    def printer(self, nodes: List[Node] = None) -> str:
        if nodes is None:
            nodes = self.get_leaves()
        for node in nodes:
            node.printer()

    def __str__(self) -> str:
        s = ""
        for node in self.get_leaves():
            s += node.__str__() + "\n"
        return s

    def volume(self, nodes: List[Node] = None) -> float:
        if nodes is None:
            nodes = self.get_leaves()
        vol = 0
        for node in nodes:
            vol += node.volume()
        return vol

    def area(self, nodes: List[Node] = None) -> float:
        if nodes is None:
            nodes = self.get_leaves()
        area = 0
        for node in nodes:
            area += node.area()
        return area
