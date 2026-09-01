import numpy as np


class DataLeaf:
    """Legacy leaf wrapper carrying event data and its rate."""

    def __init__(self, parent, data=None, idx=None):
        """Create a data leaf associated with an optional parent node."""
        self.parent = parent
        self.children = children
        self.data = data
        self.idx = idx

    @property
    def rate(self):
        """Return the rate exposed by the stored data object."""
        return self.data.rate

    def reinitialize_rates(self):
        """Provide the legacy rate-refresh hook for tree compatibility."""
        pass

    @property
    def empty(self):
        """Return whether the leaf currently stores no data."""
        if self.data == None:
            return True
        else:
            return False


class SumNode:
    """Legacy internal sum-tree node aggregating child rates."""

    def __init__(self, parent, children: set = set()):
        """Create a node with a parent and child collection."""
        self.parent = parent
        self.children = children
        self.rate = None

    def set_rate(self):
        """Set this node's rate to the sum of child rates."""
        for child in self.children:
            self.rate += child.rate

    def reinitialize_rates(self):
        """Refresh child rates recursively and then this node's rate."""
        for child in self.children:
            child.reinitialize_rates()
        self.set_rate()

    def update_rate(self):
        """Refresh this node and propagate the update to its parent."""
        self.set_rate()
        self.parent.update_rate()


class DynamicSumTree:
    """Resizable binary sum tree for weighted event sampling."""

    write = 0
    n_entries = 0

    def __init__(self, capacity=1, verbose=0):
        """Create a tree with a power-of-two leaf capacity."""
        capacity = 2 ** (np.ceil(np.log2(capacity)).astype("int"))

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.available_entries = set(range(capacity))
        self.taken_entries = set()

        self.verbose = verbose

    def _propagate(self, idx, change):
        """Apply a leaf-rate change to all ancestor nodes."""
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent > 0:
            self._propagate(parent, change)

    def _rebuild(self, idx):
        """Rebuild parent sums upward from a tree node."""
        if idx < (self.capacity - 1):
            left = 2 * idx + 1
            right = left + 1

            self.tree[idx] = self.tree[left] + self.tree[right]

        parent = (idx - 1) // 2
        if parent > 0:
            self._rebuild(parent)

    def _retrieve(self, idx, s):
        """Find the leaf whose cumulative interval contains ``s``."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def double_size(self):
        """Double leaf capacity while retaining current tree entries."""
        self.data = np.concatenate((self.data, np.zeros(self.capacity, dtype=object)))

        self.available_entries |= set(range(self.capacity, 2 * self.capacity, 1))

        self.tree = np.concatenate((self.tree, np.zeros(2 * self.capacity)))
        self.tree[(-2) * self.capacity : (-1) * self.capacity] = self.tree[
            (-3) * self.capacity : (-2) * self.capacity
        ]

        self.capacity *= 2

        for i in range(1 * self.capacity - 1, 2 * self.capacity - 1, 2):
            self._rebuild(i)

    def total(self):
        """Return the total rate of all active entries."""
        return self.tree[0]

    def add(self, p, data):
        """Insert ``data`` with rate ``p`` and return its data index."""
        try:
            idx = self.available_entries.pop()
        except KeyError:
            self.double_size()
            idx = self.available_entries.pop()

        self.data[idx] = data
        self.update(idx, p)

        self.taken_entries.add(idx)

        self.n_entries += 1
        return idx

    def update(self, idx, p):
        """Replace the rate at data index ``idx`` with ``p``."""
        idx = idx + self.capacity - 1
        change = p - self.tree[idx]

        self.tree[idx] = p

        if idx > 0:
            self._propagate(idx, change)

    def change(self, idx, data):
        """Replace the data payload at ``idx`` without changing its rate."""
        self.data[idx] = data

    def delete(self, idx):
        """Remove the entry at ``idx`` and return its slot to the free pool."""
        self.change(idx, 0)
        self.update(idx, 0)
        self.n_entries -= 1
        self.taken_entries.remove(idx)
        self.available_entries.add(idx)

    def get(self, s):
        """Return the entry selected by cumulative rate coordinate ``s``."""
        assert s <= self.total(), "Probability is too high"

        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (dataIdx, self.tree[idx], self.data[dataIdx])

    def pop(self, s):
        """Return and remove the entry selected by cumulative coordinate ``s``."""
        item = self.get(s)
        self.delete(item[0])
        return out

    def __repr__(self):
        """Return a compact representation of tree state."""
        string = "DynamicSumTree("
        for key, val in self.__dict__.items():
            if key not in ["lat", "et"]:
                string += f"{key}={val}, "
            else:
                string += f"{key}=(...), "
        string += ")"
        return string
