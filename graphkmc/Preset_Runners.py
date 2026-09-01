import math
import os
import random
import time

import numpy as np

from .Crystals import *
from .Events import EventTable
from .Lattice import Lattice, OLEDPoint, Point
from .PostProcessors import *
from .Processes import (
    ConstanteRate,
    CustomProcess,
    DexterRate,
    ForsterRate,
    KappaFactor,
    ProcessTable,
)
from .RateSampler import Rate_Sampler, SimulationTimer
from .Run import Run
from .SumTree import DynamicSumTree


def touch(fname):
    """Create ``fname`` if absent, or update its modification timestamp."""
    try:
        os.utime(fname, None)
    except OSError:
        open(fname, "a").close()


lattice_dict = {"scc": SCC_cell, "bcc": BCC_cell, "fcc": FCC_cell}


class SteadyStateRunnerTTA:
    """Preset builder and runner for steady-state triplet-triplet annihilation."""

    def get_defaults(self):
        """Return default steady-state TTA configuration values."""
        defa = dict()
        defa["verbose"] = 0
        defa["seed"] = random.randint(-10000, 10000)
        defa["name"] = "Run1"
        defa["n"] = 11
        defa["kr"] = 1.0
        defa["Rf"] = 2.0
        defa["a"] = 1.0
        defa["G"] = 1.0
        defa["max_steps"] = np.inf
        defa["max_time"] = 30.0  # math.inf
        defa["data_dir"] = os.path.join(os.getcwd(), "data")
        defa["varKappa"] = True
        defa["latticetype"] = "scc"

        try:
            defa["jobid"] = os.environ["SLURM_JOB_ID"]
        except:
            defa["jobid"] = None
        defa["ops"] = []
        defa["wait_till_finished"] = True
        return defa

    def __init__(self, **kwargs):
        """Build a configured lattice, process table, trackers, and KMC run."""
        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__
        os.makedirs(self.data_dir, exist_ok=True)
        self.find_density_file()
        self.setup_post_processors()
        self.setup_lattice()
        self.initialize_state()
        self.setup_process_table()

        self.runner = Run(**self.__dict__)

    @property
    def rcutRf(self):
        """Return the Förster interaction cutoff derived from ``Rf``."""
        return max(
            [(self.Rf * self.Rf / 1.0) * 9.1 / ((3.0 * 3.0) / 1.0), self.Rf * 3.0, 2.1]
        )

    @property
    def jobname(self):
        """Return the parameterized output-name stem for this run."""
        return f"SS_TTA_{self.varKappa}kappa_{self.latticetype}_n{self.n}kr{self.kr:.1f}Rf{self.Rf:.2f}rcut{self.rcutRf:.1f}a{self.a:.1f}G{self.G:.2e}id{self.jobid}"

    def run(self):
        """Execute the configured run and optionally wait for tracker output."""
        self.runner.run()

        if self.wait_till_finished:
            while not self.denstracker.write_queue.empty():
                time.sleep(1.0)

    def find_density_file(self):
        """Choose a non-existing density-tracker output path."""
        i = 0
        filename = "DensityTracker_" + self.jobname + f"_{i}"
        filepath = os.path.join(self.data_dir, filename + ".txt")
        while os.path.isfile(filepath):
            i += 1
            filename = "DensityTracker_" + self.jobname + f"_{i}"
            filepath = os.path.join(self.data_dir, filename + ".txt")
        self.density_file = filepath

    def setup_post_processors(self):
        """Create the radiative-decay and particle-density trackers."""
        self.krtracker = RadiativeDecayTracker()
        self.ops.append(self.krtracker)
        self.denstracker = ParticleTracker(file_path=self.density_file)
        self.ops.append(self.denstracker)

    def setup_lattice(self):
        """Construct the selected crystal supercell and populate OLED points."""
        self.cell = lattice_dict[self.latticetype.lower()](self.a)
        self.scell = SuperCell(self.n, self.cell)
        self.lat = Lattice(unit_cell=self.scell.lattice_vectors, verbose=self.verbose)
        for coords in self.scell:
            self.lat.add_point(OLEDPoint(coords=coords, mat=0))

    def initialize_state(self):
        """Estimate and sample the initial triplet population."""

        def guess_triplet_density(G, kr, Rf, a):
            """Estimate steady-state density from generation and transfer scales."""
            if (G / kr) >= max(
                1.0, (3 / (2 * np.pi * np.pi * ((Rf / a) ** 3))) ** (-2)
            ):
                frac = 1.0
            elif (G / kr) <= min(
                1.0, (3 / (2 * np.pi * np.pi * ((Rf / a) ** 3))) ** (1)
            ):
                frac = G / kr
            else:
                frac = ((G / kr) ** (1.0 / 3)) * (
                    (2 * np.pi * np.pi * ((Rf / a) ** 3)) ** (-2.0 / 3)
                )
            return frac

        self.triplet_density = guess_triplet_density(self.G, self.kr, self.Rf, self.a)

        self.lat.initialise_state()
        for point in self.lat.points:
            point.state = 0
            if random.random() <= self.triplet_density:
                point.state = 1

    def setup_process_table(self):
        """Register annihilation, decay, and generation processes."""
        self.pt = ProcessTable()

        inits = tuple([1, 1])
        fins = tuple([1, 0])
        mats = tuple([0, 0])
        rcut = self.rcutRf
        if self.varKappa:
            self.kappaf = KappaFactor(math.sqrt(3.0 / 2))
        else:
            self.kappaf = 1.0
        forsterrate = ForsterRate(self.Rf, kr=self.kr, kappa=self.kappaf)
        proc = CustomProcess(inits, fins, mats, self.rcutRf, rate=forsterrate)
        self.pt.add_possible_process(proc)
        self.denstracker.neg_proc_ids.append(proc.proc_idx)

        inits = tuple([1])
        fins = tuple([0])
        mats = tuple([0])
        rcut = 0.01
        proc = CustomProcess(inits, fins, mats, rcut, rate=self.kr)
        self.pt.add_possible_process(proc)
        self.krtracker.proc_ids.append(proc.proc_idx)
        self.denstracker.states.append(1)
        self.denstracker.neg_proc_ids.append(proc.proc_idx)

        inits = tuple([0])
        fins = tuple([1])
        mats = tuple([0])
        rcut = 0.01
        proc = CustomProcess(inits, fins, mats, rcut, rate=self.G)
        self.pt.add_possible_process(proc)
        self.denstracker.pos_proc_ids.append(proc.proc_idx)


class Charge_Transport:
    """Incomplete preset scaffold for charge-transport simulations."""

    def get_defaults(self):
        """Return default charge-transport configuration values."""
        defa = dict()
        defa["verbose"] = 0
        defa["seed"] = random.randint(-10000, 10000)
        defa["name"] = "Run1"
        defa["n"] = 11
        defa["a"] = 1.0
        defa["nu0"] = 1.0
        defa["labda"] = 0.3
        defa["density"] = 1e-3
        defa["max_steps"] = np.inf
        defa["max_time"] = 30.0  # math.inf
        defa["data_dir"] = os.path.join(os.getcwd(), "data")
        defa["varKappa"] = True
        defa["latticetype"] = "scc"

        try:
            defa["jobid"] = os.environ["SLURM_JOB_ID"]
        except:
            defa["jobid"] = None
        defa["ops"] = []
        defa["wait_till_finished"] = True
        return defa

    def __init__(self, **kwargs):
        """Build the configured components and create a KMC run."""
        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__
        self.find_density_file()
        self.setup_post_processors()
        self.setup_lattice()
        self.initialize_state()
        self.setup_process_table()

        self.runner = Run(**self.__dict__)

    @property
    def rcutRf(self):
        """Return the Förster interaction cutoff derived from ``Rf``."""
        return max(
            [(self.Rf * self.Rf / 1.0) * 9.1 / ((3.0 * 3.0) / 1.0), self.Rf * 3.0, 2.1]
        )

    @property
    def jobname(self):
        """Return the parameterized output-name stem for this run."""
        return f"SS_TTA_{self.varKappa}kappa_{self.latticetype}_n{self.n}kr{self.kr:.1f}Rf{self.Rf:.2f}rcut{self.rcutRf:.1f}a{self.a:.1f}G{self.G:.2e}id{self.jobid}"

    def run(self):
        """Execute the configured run and optionally wait for output."""
        self.runner.run()

        if self.wait_till_finished:
            while not self.denstracker.write_queue.empty():
                time.sleep(1.0)

    def find_density_file(self):
        """Reserve a density output path; subclasses should implement this."""
        pass

    def setup_post_processors(self):
        """Create output processors; subclasses should implement this."""
        pass

    def setup_lattice(self):
        """Construct the selected crystal supercell and populate OLED points."""
        self.cell = lattice_dict[self.latticetype.lower()](self.a)
        self.scell = SuperCell(self.n, self.cell)
        self.lat = Lattice(unit_cell=self.scell.lattice_vectors, verbose=self.verbose)
        for coords in self.scell:
            self.lat.add_point(OLEDPoint(coords=coords, mat=0))

    def initialize_state(self):
        """Sample initial carrier states from the configured triplet density."""

        self.lat.initialise_state()
        for point in self.lat.points:
            point.state = 0
            if random.random() <= self.triplet_density:
                point.state = 1

    def setup_process_table(self):
        """Register transport, decay, and generation processes."""
        self.pt = ProcessTable()

        inits = tuple([1, 1])
        fins = tuple([1, 0])
        mats = tuple([0, 0])
        rcut = self.rcutRf
        if self.varKappa:
            self.kappaf = KappaFactor(math.sqrt(3.0 / 2))
        else:
            self.kappaf = 1.0
        forsterrate = ForsterRate(self.Rf, kr=self.kr, kappa=self.kappaf)
        proc = CustomProcess(inits, fins, mats, self.rcutRf, rate=forsterrate)
        self.pt.add_possible_process(proc)
        self.denstracker.neg_proc_ids.append(proc.proc_idx)

        inits = tuple([1])
        fins = tuple([0])
        mats = tuple([0])
        rcut = 0.01
        proc = CustomProcess(inits, fins, mats, rcut, rate=self.kr)
        self.pt.add_possible_process(proc)
        self.krtracker.proc_ids.append(proc.proc_idx)
        self.denstracker.states.append(1)
        self.denstracker.neg_proc_ids.append(proc.proc_idx)

        inits = tuple([0])
        fins = tuple([1])
        mats = tuple([0])
        rcut = 0.01
        proc = CustomProcess(inits, fins, mats, rcut, rate=self.G)
        self.pt.add_possible_process(proc)
        self.denstracker.pos_proc_ids.append(proc.proc_idx)
