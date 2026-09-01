"""Numba-compiled vector and distance operations."""

import math

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def dot(v1, v2):
    """Return the dot product of two Numba-compatible vectors."""
    return sum([v1i * v2i for v1i, v2i in zip(v1, v2)])


@nb.njit(fastmath=True)
def distance_vector_nb(xyz1, xyz2, boxsize):
    """Return the minimum-image displacement under periodic boundaries."""
    out = np.zeros_like(xyz1)
    for i in range(len(xyz1)):
        out[i] = xyz2[i] - xyz1[i]
        out[i] = (out[i] + 0.5 * boxsize[i]) % boxsize[i] - 0.5 * boxsize[i]
    return out


@nb.njit(fastmath=True)
def distance_sq_nb(xyz1, xyz2, boxsize):
    """Return squared minimum-image distance in a periodic box."""
    displacement = distance_vector_nb(xyz1, xyz2, boxsize)
    return sum(displacement * displacement)


@nb.njit(fastmath=True)
def distance_nb(xyz1, xyz2, boxsize):
    """Return minimum-image distance in a periodic box."""
    return math.sqrt(distance_sq_nb(xyz1, xyz2, boxsize))


@nb.njit(fastmath=True)
def invdistance_nb(xyz1, xyz2, boxsize):
    """Return reciprocal minimum-image distance in a periodic box."""
    return 1.0 / distance_nb(xyz1, xyz2, boxsize)
