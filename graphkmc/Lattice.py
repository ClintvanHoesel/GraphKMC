from __future__ import annotations

import math
import random
import warnings

import numpy as np
import scipy
import scipy.spatial
from scipy.spatial import KDTree
from tqdm import tqdm

from .utils import distance_nb, distance_sq_nb, distance_vector_nb, invdistance_nb


class Point:
    """A lattice site with material, state, neighbours, and process links."""

    def get_defaults(self):
        """Return default point attributes."""
        defa = dict()
        defa["mat"] = 0
        defa["single_procs"] = []  # available processes for this material
        defa["active_procs"] = []  # available processes for this material
        defa["undergone_procs"] = []  # available processes for this material
        defa["dim"] = 3
        defa["neighbours"] = []
        defa["proc_neighbours"] = dict()
        defa["events"] = []
        defa["state"] = 0
        defa["verbose"] = 0

        return defa

    def __init__(self, **kwargs):
        """Create a point; ``coords`` must be supplied in the lattice dimension."""
        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

        assert hasattr(self, "coords")
        assert len(self.coords) == self.dim

    def initialise_proc_neighbours(self):
        """Build per-process neighbour sets that lie within each cutoff."""
        for proc in self.single_procs:
            if proc not in self.proc_neighbours.keys():
                self.proc_neighbours[proc] = set()

        for neighbour in self.neighbours:
            dist_sq = self.distance_sq(neighbour)
            for proc in set(self.active_procs + self.undergone_procs):
                if proc not in self.proc_neighbours.keys():
                    self.proc_neighbours[proc] = set()
                if dist_sq <= proc.rcutsq:
                    if proc in neighbour.undergone_procs:
                        self.proc_neighbours[proc].add(neighbour)

    def distance_vector(self, other: Point):
        """Return the minimum-image displacement to ``other``."""
        return distance_vector_nb(self.coords, other.coords, self.lat.boxsize)

    def distance_sq(self, other: Point):
        """Return squared minimum-image distance to ``other``."""
        return distance_sq_nb(self.coords, other.coords, self.lat.boxsize)

    def distance(self, other: Point):
        """Return minimum-image distance to ``other``."""
        return distance_nb(self.coords, other.coords, self.lat.boxsize)

    def process_state_change(self, new_state):
        """Change state and rebuild all events affected by this point."""
        self.state = new_state
        self.remove_events()

        self.et.add_single_particle_events(self)
        self.et.add_active_particle_events(self)
        self.et.add_undergone_particle_events(self)

    def remove_events(self):
        """Remove every active event involving this point."""
        while len(self.events) > 0:
            self.events[0].remove()

    def __repr__(self):
        """Return a representation that omits cyclic lattice references."""
        string = "Point("
        for key, val in self.__dict__.items():
            if key not in ["lat", "events", "neighbours", "et", "proc_neighbours"]:
                string += f"{key}={val}, "
            else:
                string += f"{key}=(...), "
        string += ")"
        return string


class OLEDPoint(Point):
    """Point with randomly generated molecular energies and dipole vectors."""

    def get_defaults(self):
        """Extend point defaults with OLED energetic and dipole properties."""
        defa = super().get_defaults()
        defa["mu_HOMO"] = 0.0
        defa["sigma_HOMO"] = 0.1
        defa["mu_LUMO"] = 0.0
        defa["sigma_LUMO"] = 0.1
        defa["E_LUMO"] = None
        defa["E_HOMO"] = None
        defa["thetamu"] = None
        defa["phimu"] = None
        defa["mu"] = None
        defa["thetamuA"] = None
        defa["phimuA"] = None
        defa["muA"] = None
        defa["thetamuD"] = None
        defa["phimuD"] = None
        defa["muD"] = None

        return defa

    def __init__(self, **kwargs):
        """Initialize an OLED point, generating omitted energies and dipoles."""
        super().__init__(**kwargs)
        if self.mu == None:
            self.mu = self.generate_mu(self.thetamu, self.phimu)
        if self.muA == None:
            self.muA = self.generate_mu(self.thetamuA, self.phimuA)
        if self.muD == None:
            self.muD = self.generate_mu(self.thetamuD, self.phimuD)
        if self.E_HOMO == None:
            self.E_HOMO = self.generate_guassian(self.mu_HOMO, self.sigma_HOMO)
        if self.E_LUMO == None:
            self.E_LUMO = self.generate_guassian(self.mu_LUMO, self.sigma_LUMO)

    def generate_theta(self):
        """Sample a polar angle for an isotropically distributed dipole."""
        theta = random.uniform(0.0, 1.0)
        return math.acos(1 - 2 * theta)

    def generate_phi(self):
        """Sample an azimuthal angle for an isotropically distributed dipole."""
        return random.uniform(0.0, 2 * math.pi)

    def generate_mu(self, theta=None, phi=None):
        """Return a unit dipole vector for supplied or sampled angles."""
        if theta == None:
            theta = self.generate_theta()

        if phi == None:
            phi = self.generate_phi()

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        mu = [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
        return mu

    def generate_guassian(self, mu=0.0, sigma=1.0):
        """Sample a Gaussian-distributed molecular energy value."""
        return random.gauss(mu, sigma)


class Lattice:
    """Periodic orthogonal lattice containing points and neighbourhood data."""

    def get_defaults(self):
        """Return default lattice attributes."""
        defa = dict()
        defa["nstates"] = 1
        defa["verbose"] = 0
        defa["dim"] = 3
        defa["edges"] = []

        return defa

    def __init__(self, **kwargs):
        """Create a lattice from orthogonal ``unit_cell`` vectors."""
        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

        assert hasattr(self, "unit_cell")
        assert len(self.unit_cell) == self.dim
        assert all(len(v) == self.dim for v in self.unit_cell)
        for i in range(self.dim):
            assert max(self.unit_cell[i]) / sum(self.unit_cell[i]) >= 1.0
            for j in range(i + 1, self.dim):
                assert (
                    sum(
                        self.unit_cell[i][k] * self.unit_cell[j][k]
                        for k in range(self.dim)
                    )
                    == 0
                )
        self.boxsize = np.diag(
            self.unit_cell
        )  # [self.unit_cell[i][i] for i in range(self.dim)]#np.diag(self.unit_cell)

        if not hasattr(self, "points"):
            self.points = []

        self.initialized = False

    def initialize(self):
        """Finalize point links, neighbours, and state for one-time use."""
        if self.initialized:
            raise ValueError("Lattice is already initialized!")
        self.set_lattice_points()
        self.set_lattice_idx_points()

        self.initialise_neighbours()

        if not hasattr(self, "state"):
            self.initialise_state()

        self.initialized = True

    def add_point(self, point: Point):
        """Add ``point`` to this lattice and assign its lattice index."""
        point.li = len(self.points)
        point.lat = self
        self.points.append(point)

    def set_lattice_points(self):
        """Attach this lattice to every contained point."""
        for point in self.points:
            point.lat = self

    def set_lattice_idx_points(self):
        """Refresh the lattice index stored on each point."""
        for idx, point in enumerate(self.points):
            point.li = idx

    def initialise_state(self, fill=None):
        """Optionally set every point's state to ``fill``."""
        if not (fill == None):
            for p in self.points:
                p.state = fill

    @property
    def state(self):
        """Return current states in lattice-point order."""
        return [p.state for p in self.points]

    @property
    def density(self):
        """Return the number density of lattice points."""
        return len(self.points) / math.prod(self.boxsize)

    @property
    def max_r_cutoff(self):
        """Return the largest cutoff registered in the process table."""
        return self.pt.max_r_cutoff

    def initialise_neighbours(self):
        """Find periodic geometric neighbours within the largest process cutoff."""
        xyzs = [point.coords for point in self.points]
        self.kdtree = KDTree(np.array(xyzs), boxsize=self.boxsize)
        if self.verbose > 3:
            itt = tqdm(self.points, desc="Neighbours")
        else:
            itt = self.points
        for p1 in itt:
            neighs = self.kdtree.query_ball_point(
                p1.coords, self.max_r_cutoff, return_sorted=False
            )
            for i_p2 in neighs:
                p2 = self.points[i_p2]
                if p1 == p2:
                    continue
                p1.neighbours.append(p2)

    def initialise_proc_neighbours(self):
        """Build process-specific neighbour sets for every point."""
        if self.verbose > 3:
            itt = tqdm(self.points, desc="Proc neighbours")
        else:
            itt = self.points
        for p1 in itt:
            p1.initialise_proc_neighbours()

    def __repr__(self):
        """Return a representation of the lattice configuration."""
        string = "Lattice("
        for key, val in self.__dict__.items():
            string += f"{key}={val}, "
        string += ")"
        return string
