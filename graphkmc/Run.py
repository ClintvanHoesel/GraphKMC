import dbm
import math
import os
import queue
import random
import struct
from threading import Lock, Thread

from tqdm import tqdm

from .Events import EventTable
from .Lattice import Lattice, Point
from .Processes import CustomProcess, ProcessTable
from .RateSampler import Rate_Sampler, SimulationTimer
from .SumTree import DynamicSumTree


class Run:
    """Coordinate lattice, processes, event sampling, and output processors."""

    def get_defaults(self):
        """Return default KMC-run configuration values."""
        defa = dict()
        defa["max_steps"] = math.inf
        defa["max_time"] = math.inf
        defa["n_steps"] = 0
        defa["verbose"] = 0
        defa["seed"] = random.randint(-10000, 10000)
        defa["name"] = "Run1"
        return defa

    def __init__(self, lat: Lattice, pt: ProcessTable, ops=None, **kwargs):
        """Initialize a runnable KMC simulation on ``lat`` using ``pt``."""
        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

        self.lat = lat

        self.pt = pt

        self.st = DynamicSumTree(len(self.lat.points))

        self.simtim = SimulationTimer(verbose=self.verbose)
        self.rs = Rate_Sampler(self.st, seed=self.seed, verbose=self.verbose)

        self.pt.register_lattice(self.lat)

        self.pt.initialize_available_procs_per_point()
        self.lat.initialise_neighbours()
        self.lat.initialise_proc_neighbours()

        self.et = EventTable(self.lat, self.st, verbose=self.verbose)
        self.et.initialise_events()

        self.ops = [] if ops is None else list(ops)
        self.register_at_ops()

    def reset(self, new_name=None, init_time=0.0, reinit_events=True):
        """Reset time and optionally event state for another realization."""
        if new_name is not None:
            self.name = new_name
            self.register_at_ops()

        if reinit_events:
            while len(self.et.events) > 0:
                self.et.events[0].remove()

            self.et.initialise_events()

        self.simtim.time = init_time

    def step(self):
        """Sample, record, and apply one KMC event."""
        chosen_event, dt = self.rs.sample()

        for op in self.ops:
            op(self, chosen_event, dt, self.simtim.time)

        self.simtim.update_time(dt)
        self.et.process_event(chosen_event[2])
        self.n_steps += 1
        return chosen_event

    def register_at_ops(self):
        """Register this run with every configured output processor."""
        for op in self.ops:
            op.register_run(self)

    def continue_criteria(self):
        """Return whether events and configured time/step limits permit progress."""
        criteria = []
        criteria.append(len(self.et) > 0)
        criteria.append(self.simtim.time < self.max_time)
        criteria.append(self.n_steps < self.max_steps)
        return all(criteria)

    def run(self):
        """Execute KMC steps until no event or configured limit remains."""
        pbar = tqdm(
            desc=self.name,
            dynamic_ncols=True,
            total=self.max_steps,
            initial=self.n_steps,
            mininterval=1.0,
        )
        while self.continue_criteria():
            self.step()
            pbar.update()
        pbar.close()


from .PostProcessors import JumpTracker  # noqa: E402,F401
from .PostProcessors import (
    MultiVerseJumpTracker,
    MultiVerseJumpTracker_v3,
    ParticleTracker,
    RadiativeDecayTracker,
)
