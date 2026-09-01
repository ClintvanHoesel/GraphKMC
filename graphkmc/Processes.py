import warnings

from .Lattice import Lattice, Point
from .SumTree import DynamicSumTree
from .utils import dot


class BaseRateCalculator:
    """Base callable interface for process-rate calculations."""

    def get_defaults(self):
        """Return default rate-calculator configuration values."""
        defa = dict()
        return defa

    def __init__(self, **kwargs):
        """Initialize the calculator with optional configuration overrides."""
        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

    def __call__(self, points, *args, **kwargs):
        """Evaluate the rate for ``points``."""
        return self.rate(points, *args, **kwargs)

    def rate(self, points, *args, **kwargs):
        """Compute a rate; subclasses must implement this method."""
        raise NotImplementedError()


class ConstanteRate(BaseRateCalculator):
    """Rate calculator that always returns the same value."""

    def __init__(self, rate, **kwargs):
        """Store the constant process rate."""
        self.k = rate
        self.kr = self.k

    def rate(self, points, *args, **kwargs):
        """Return the configured constant rate."""
        return self.k


class KappaFactor(BaseRateCalculator):
    """Orientation-dependent Förster kappa-factor calculator."""

    def __init__(self, rate, **kwargs):
        """Set the kappa-factor scale multiplier."""
        self.k = rate

    def rate(self, points, *args, **kwargs):
        """Calculate the dipole-orientation factor for two points."""
        out1 = dot(points[0].mu, points[1].mu)
        dr = points[0].distance_vector(points[1]) / points[0].distance(points[1])
        out2 = 3 * dot(points[0].mu, dr) * dot(points[1].mu, dr)
        return self.k * (out1 - out2)


class ForsterRate(BaseRateCalculator):
    """Förster transfer rate with distance, orientation, and energy factors."""

    def __init__(self, Rf, lifetime=None, kr=None, kappa=1.0, deltaEf=1.0, **kwargs):
        """Configure a Förster rate from radius and lifetime or decay rate."""
        self.Rf = Rf
        self.Rf6 = Rf**6
        if lifetime != None:
            kr = 1.0 / lifetime
            self.lifetime = lifetime
            self.kr = kr
        elif kr != None:
            lifetime = 1.0 / kr
            self.lifetime = lifetime
            self.kr = kr
        else:
            raise ValueError(f"Did not get lifetime nor kr.")

        if not callable(kappa):
            warnings.warn(f"kappasq not callable!")
            kappa = ConstanteRate(kappa)
        self.kappa = kappa

        if not callable(deltaEf):
            warnings.warn(f"kappasq not callable!")
            deltaEf = ConstanteRate(deltaEf)
        self.deltaEf = deltaEf

        self.prefactor = self.kr * self.Rf6

        super().__init__(**kwargs)

    def rate(self, points, *args, **kwargs):
        """Calculate the Förster transfer rate between two points."""
        distsq = points[0].distance_sq(points[1])
        inv_dist6 = 1.0 / ((distsq**3))
        kappasq = (self.kappa(points, *args, **kwargs)) ** 2
        rho_deltaE = self.deltaEf(points, *args, **kwargs)
        return self.prefactor * kappasq * inv_dist6 * rho_deltaE


class DexterRate(BaseRateCalculator):
    """Dexter transfer rate with exponential distance decay."""

    def __init__(self, nu0, labda=0.3, deltaEf=1.0, **kwargs):
        """Configure prefactor, decay length, and energy-overlap factor."""
        self.nu0 = nu0
        self.labda = labda
        self.twoinv_labda = -2.0 / self.labda

        if not callable(deltaEf):
            warnings.warn(f"kappasq not callable!")
            deltaEf = ConstanteRate(deltaEf)
        self.deltaEf = deltaEf

        super().__init__(**kwargs)

    def rate(self, points, *args, **kwargs):
        """Calculate the Dexter transfer rate between two points."""
        dist = points[0].distance(points[1])
        rho_deltaE = self.deltaEf(points, *args, **kwargs)
        return self.nu0 * np.exp(-2.0 * dist / self.labda) * rho_deltaE


class CustomProcess:
    """State transition allowed for one or two points within a cutoff radius."""

    def __init__(
        self,
        init_state: tuple,
        final_state: tuple,
        materials: tuple,
        rcut: float,
        rate,
        callbacks=[],
    ):
        """Define a process by initial/final states, materials, cutoff, and rate."""
        assert len(init_state) == len(final_state) == len(materials)
        assert len(init_state) <= 2

        self.init_state = init_state
        self.final_state = final_state
        self.materials = materials
        self.rcut = rcut
        self.rcutsq = rcut * rcut

        self.state_change = tuple(
            fst != ist for ist, fst in zip(self.init_state, self.final_state)
        )

        self.callbacks = callbacks

        if not callable(rate):
            warnings.warn(f"Rate not callable!")
            rate = ConstanteRate(rate)
        self.rate = rate

    def __call__(self, *args, **kwargs):
        """Evaluate this process's configured rate calculator."""
        return self.rate(*args, **kwargs)

    def __repr__(self):
        """Return a representation of the process configuration."""
        string = "Process("
        for key, val in self.__dict__.items():
            string += f"{key}={val},"
        string += ")"
        return string

    @property
    def n_particles(self):
        """Return the number of points participating in this process."""
        return len(self.materials)

    def process(self, *args, points: tuple, **kwargs):
        """Apply this process's state changes to ``points``."""
        for point, change, new_state in zip(
            points, self.state_change, self.final_state
        ):
            if change:
                point.process_state_change(new_state)


class ProcessTable:
    """Registry of processes and their allowed lattice roles."""

    def get_defaults(self):
        """Return default process-table storage."""
        defa = dict()
        defa["procs"] = []
        defa["states"] = set()
        return defa

    def __init__(self, lat=None, **kwargs):
        """Create an empty process table, optionally attached to ``lat``."""

        self.__dict__.update(kwargs)
        if lat is not None:
            self.lat = lat
        self.__dict__ = self.get_defaults() | self.__dict__

        if hasattr(self, "lat"):
            self.register_lattice(self.lat)

    def add_possible_process(self, proc: CustomProcess):
        """Assign an index to and register a possible process."""
        proc.proc_idx = self.n_procs
        self.procs.append(proc)

        for state in proc.init_state:
            self.states.add(state)
        for state in proc.final_state:
            self.states.add(state)

    @property
    def n_procs(self):
        """Return the number of registered processes."""
        return len(self.procs)

    def nstates(self):
        """Return the number of states referenced by registered processes."""
        return len(self.states)

    def get_rcutoffs(self):
        """Return pairs of process indices and their cutoff radii."""
        return [(proc.proc_idx, proc.rcut) for proc in self.procs]

    def register_lattice(self, lat: Lattice):
        """Attach this process table to ``lat``."""
        self.lat = lat
        self.lat.pt = self

    @property
    def max_r_cutoff(self):
        """Return the largest process interaction cutoff."""
        return max(self.get_rcutoffs(), key=lambda item: item[1])[1]

    def initialize_available_procs_per_point(self):
        """Classify processes available to each lattice point by material role."""
        for point in self.lat.points:
            for proc in self.procs:
                if point.mat in proc.materials:
                    if len(proc.materials) == 1:
                        point.single_procs.append(proc)

                    if len(proc.materials) == 2:
                        if point.mat == proc.materials[0]:
                            point.active_procs.append(proc)

                        if point.mat == proc.materials[1]:
                            point.undergone_procs.append(proc)

    def __repr__(self):
        """Return a compact representation of the process table."""
        string = "ProcessTable("
        for key, val in self.__dict__.items():
            if key == "lat":
                string += f"{key}=Lat,"
            else:
                string += f"{key}={val},"
        string += ")"
        return string
