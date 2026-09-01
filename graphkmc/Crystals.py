import itertools

import numpy as np


class BaseCell:
    """Abstract unit cell describing basis points and lattice vectors."""

    def __init__(self, *args, **kwargs):
        """Initialize a unit-cell base class."""
        pass

    def _points(self):
        """Return the unit-cell basis coordinates."""
        raise NotImplementedError()

    @property
    def points(self):
        """Return the coordinates of points in the unit-cell basis."""
        return self._points()

    def _lat_vecs(self):
        """Return the lattice vectors defining the unit cell."""
        raise NotImplementedError()

    @property
    def npoints(self):
        """Return the number of basis points in the cell."""
        return len(self.points)

    @property
    def lattice_vectors(self):
        """Return the unit-cell lattice vectors."""
        return self._lat_vecs()

    @property
    def dim(self):
        """Return the spatial dimension of the cell."""
        return self.lattice_vectors.shape[1]

    def __iter__(self):
        """Iterate over basis-point coordinates."""
        for p in self.points:
            yield p


class SCC_cell(BaseCell):
    """Simple cubic unit cell with one point at the origin."""

    def __init__(self, a=1.0, *args, **kwargs):
        """Create a simple cubic cell with lattice scale ``a``."""
        self.a = a
        super().__init__(*args, **kwargs)

    def _points(self):
        """Return the simple-cubic basis point."""
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    def _lat_vecs(self):
        """Return orthogonal simple-cubic lattice vectors."""
        return (
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
            * self.a
        )


class FCC_cell(BaseCell):
    """Face-centred cubic unit cell with a four-point basis."""

    def __init__(self, a=1.0, b=None, *args, **kwargs):
        """Create an FCC cell from density scale ``a`` or edge length ``b``."""
        assert a or b
        if b:
            a = b * ((4.0) ** (-1.0 / 3))
        self.a = a
        super().__init__(*args, **kwargs)

    @property
    def b(self):
        """Return the conventional cubic edge length."""
        return self.a * ((4.0) ** (1.0 / 3))

    def _points(self):
        """Return the four FCC basis points."""
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [self.b / 2, self.b / 2, 0.0],
                [self.b / 2, 0.0, self.b / 2],
                [0.0, self.b / 2, self.b / 2],
            ],
            dtype=np.float32,
        )

    def _lat_vecs(self):
        """Return the conventional FCC cubic lattice vectors."""
        return (
            np.array(
                [[self.b, 0.0, 0.0], [0.0, self.b, 0.0], [0.0, 0.0, self.b]],
                dtype=np.float32,
            )
            * self.a
        )


class BCC_cell(BaseCell):
    """Body-centred cubic unit cell with a two-point basis."""

    def __init__(self, a=1.0, b=None, *args, **kwargs):
        """Create a BCC cell from density scale ``a`` or edge length ``b``."""
        assert a or b
        if b:
            a = b * ((2.0) ** (-1.0 / 3))
        self.a = a
        super().__init__(*args, **kwargs)

    @property
    def b(self):
        """Return the conventional cubic edge length."""
        return self.a * ((2.0) ** (1.0 / 3))

    def _points(self):
        """Return the two BCC basis points."""
        return np.array(
            [[0.0, 0.0, 0.0], [self.b / 2, self.b / 2, self.b / 2]], dtype=np.float32
        )

    def _lat_vecs(self):
        """Return the conventional BCC cubic lattice vectors."""
        return (
            np.array(
                [[self.b, 0.0, 0.0], [0.0, self.b, 0.0], [0.0, 0.0, self.b]],
                dtype=np.float32,
            )
            * self.a
        )


class SuperCell:
    """Periodic repetition of a base cell along each lattice direction."""

    def __init__(self, n_arr, base_cell):
        """Create a supercell with ``n_arr`` repeats of ``base_cell``."""
        assert isinstance(n_arr, (int, np.array, list, tuple))
        if isinstance(n_arr, int):
            n_arr = [n_arr] * base_cell.dim
        n_arr = np.array(n_arr)
        assert n_arr.ndim == 1
        self.n_arr = n_arr
        self.base_cell = base_cell

    @property
    def lattice_vectors(self):
        """Return lattice vectors spanning the complete supercell."""
        return self.base_cell.lattice_vectors * self.n_arr[None, :]

    def __iter__(self):
        """Iterate over every translated basis point in the supercell."""
        for p in self.base_cell:
            for n_arr in itertools.product(*[range(n) for n in self.n_arr]):
                trans_vec = np.sum(
                    self.base_cell.lattice_vectors * np.array(n_arr)[None, :], axis=0
                )
                yield p + trans_vec
