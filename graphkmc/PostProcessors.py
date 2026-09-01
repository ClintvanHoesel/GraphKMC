import dbm

import h5py

try:
    import hdfdict
except ImportError:  # Optional unless HDF5-backed processors are used.
    hdfdict = None
import math
import os
import queue
import random
import sqlite3
import struct
from threading import Lock, Thread

from .Events import EventTable
from .Lattice import Lattice, Point
from .Processes import CustomProcess, ProcessTable
from .RateSampler import Rate_Sampler, SimulationTimer
from .SumTree import DynamicSumTree


class BaseOutputProcessor:
    """Base interface for components that observe and persist KMC output."""

    def get_defs(self):
        """Return default run-registration and event storage."""
        defa = dict()
        defa["runners"] = []
        defa["run_classes"] = dict()
        defa["events"] = dict()
        return defa

    def __init__(self, **kwargs):
        """Initialize the processor with optional storage overrides."""
        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defs() | self.__dict__

    def register_run(self, run):
        """Register ``run`` and create its event history."""
        self.runners.append(run.name)
        self.run_classes[run.name] = run
        self.events[run.name] = []

    def __call__(self, state, event, dt, time):
        """Record an event; subclasses implement their own output format."""
        raise NotImplementedError()


class RadiativeDecayTracker(BaseOutputProcessor):
    """Record radiative-decay events selected by process identifier."""

    def get_defaults(self):
        """Return default tracked radiative process identifiers."""
        defa = dict()
        defa["proc_ids"] = []
        return defa

    def __init__(self, **kwargs):
        """Initialize a radiative-decay tracker."""
        super().__init__()

        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

    def __call__(self, run, data, dt, time):
        """Append the selected event's radiative-decay metadata."""
        event = data[2]
        if event.proc.proc_idx in self.proc_ids:
            self.events[run.name].append(
                (
                    time,
                    dt,
                    event.points[0].coords,
                    event.points[0].mat,
                    event.proc.proc_idx,
                )
            )
        else:
            self.events[run.name].append((time, dt, None, None, None))


class ParticleTracker(BaseOutputProcessor):
    """Track selected particle populations and write changes asynchronously."""

    def get_defaults(self):
        """Return default particle-tracking configuration and storage."""
        defa = dict()
        defa["states"] = []
        defa["neg_proc_ids"] = []
        defa["pos_proc_ids"] = []
        defa["densities"] = dict()
        defa["boxsize"] = dict()
        defa["invboxsize"] = dict()
        defa["file_path"] = os.path.join(os.getcwd(), "particletracker.txt")
        defa["write_queue"] = queue.Queue()
        return defa

    def __init__(self, **kwargs):
        """Create the output file and start the asynchronous writer."""
        super().__init__()

        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

        self.output_file = open(self.file_path, "x")
        self.output_file.close()
        self.file_lock = Lock()
        self.t = Thread(target=self.write_loop, daemon=True)
        self.t.start()

    def register_run(self, run):
        """Register a run and record its initial tracked-particle count."""
        self.runners.append(run.name)
        self.run_classes[run.name] = run
        self.events[run.name] = [
            (0.0, sum(True for state in run.lat.state if state in self.states))
        ]
        self.densities[run.name] = run.lat.density
        self.boxsize[run.name] = math.prod(run.lat.boxsize)
        self.invboxsize[run.name] = 1.0 / math.prod(run.lat.boxsize)

    def write_loop(self):
        """Consume queued text records until the daemon thread exits."""
        while True:
            outstr = self.write_queue.get()
            self.write_to_file(outstr)
            self.write_queue.task_done()

    def write_to_file(self, outstr):
        """Append one text record while holding the file lock."""
        with self.file_lock:
            with open(self.file_path, "a") as f:
                f.write(outstr)

    def __call__(self, run, data, dt, time):
        """Track a population increase or decrease caused by an event."""
        event = data[2]
        if event.proc.proc_idx in self.neg_proc_ids:
            out = [
                time + dt,
                self.events[run.name][-1][1] - 1.0,
                event.proc.proc_idx,
                time,
                dt,
            ]
            self.events[run.name].append(out)
            outstr = ",".join([str(x) for x in self.events[run.name][-1]]) + "\r\n"
            self.write_queue.put(outstr)
        elif event.proc.proc_idx in self.pos_proc_ids:
            out = [
                time + dt,
                self.events[run.name][-1][1] + 1.0,
                event.proc.proc_idx,
                time,
                dt,
            ]
            self.events[run.name].append(out)
            outstr = ",".join([str(x) for x in self.events[run.name][-1]]) + "\r\n"
            self.write_queue.put(outstr)


class MultiVerseJumpTracker(BaseOutputProcessor):
    """Accumulate jump-displacement probabilities in a DBM-backed store."""

    def get_defaults(self):
        """Return default multiverse jump-tracking configuration."""
        defa = dict()
        defa["states"] = []
        defa["dim"] = 3
        defa["kr"] = 1.0
        defa["file_path"] = os.path.join(os.getcwd(), "multiverse.shelve")
        defa["distances"] = dict()
        defa["dist_time"] = dict()
        defa["proc_ids"] = []
        defa["stop_proc_ids"] = []
        defa["dist_prob"] = dict()
        defa["write_queue"] = queue.Queue()
        return defa

    def __init__(self, **kwargs):
        """Open a new DBM output store and start its writer thread."""
        super().__init__()

        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

        if os.path.isfile(self.file_path):
            raise Exception(f"{self.file_path} already exists!")
        self.output_file = dbm.open(self.file_path, "n")

        self.write_thread = Thread(target=self.write_loop, daemon=True)
        self.write_thread.start()

    def register_run(self, run):
        """Register a run with zero initial displacement."""
        super().register_run(run)
        self.distances[run.name] = tuple(0.0 for i in range(self.dim))
        self.dist_time[run.name] = list(self.distances[run.name]) + [None] + [None]

    def write_loop(self):
        """Persist queued displacement-probability contributions."""
        while True:
            curr_dist, curr_mat, add_prob = self.write_queue.get()
            self.write_to_file(curr_dist, curr_mat, add_prob)
            self.write_queue.task_done()

    def write_to_file(self, curr_dist, curr_mat, add_prob):
        """Add a probability contribution for one displacement/material bin."""
        bcurr_dist = struct.pack("dddi", *curr_dist, curr_mat)
        try:
            self.output_file[bcurr_dist] = struct.pack(
                "d", struct.unpack("d", self.output_file[bcurr_dist])[0] + add_prob
            )
        except:
            self.output_file[bcurr_dist] = struct.pack("d", add_prob)

    def __call__(self, run, data, dt, time):
        """Update displacement statistics after a KMC event."""
        event = data[2]
        if event.proc.proc_idx in self.proc_ids:
            self.distances[run.name] = tuple(
                x + y
                for x, y in zip(
                    self.distances[run.name],
                    event.points[0].distance_vector(event.points[1]),
                )
            )
        if event.proc.proc_idx in self.stop_proc_ids:
            self.dist_time[run.name] = list(self.distances[run.name]) + [time] + [dt]

        curr_mat = event.points[0].mat
        curr_dist = self.distances[run.name]
        add_prob = event.proc.rate.kr / run.st.total()

        try:
            self.dist_prob[(curr_dist, curr_mat)] += add_prob
        except:
            self.dist_prob[(curr_dist, curr_mat)] = add_prob

        self.write_queue.put((curr_dist, curr_mat, add_prob))


class JumpTracker(BaseOutputProcessor):
    """Record individual jumps and persist totals once a run stops."""

    def get_defaults(self):
        """Return default jump-tracking configuration."""
        defa = dict()
        defa["states"] = []
        defa["proc_ids"] = []
        defa["file_path"] = os.path.join(os.getcwd(), "jumptrack.shelve")
        defa["stop_proc_ids"] = []
        defa["i_run"] = 0
        return defa

    def __init__(self, **kwargs):
        """Open a new DBM output store for jump totals."""
        super().__init__()

        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

        if os.path.isfile(self.file_path):
            raise Exception(f"{self.file_path} already exists!")
        self.output_file = dbm.open(self.file_path, "n")

    def __call__(self, run, data, dt, time):
        """Record a jump or write a run total after a stopping event."""
        event = data[2]
        if event.proc.proc_idx in self.proc_ids:
            self.events[run.name].append(
                (
                    time,
                    dt,
                    event.points[0].distance_vector(event.points[1]),
                    event.proc.proc_idx,
                    event.points[1].mat,
                )
            )

        if event.proc.proc_idx in self.stop_proc_ids:
            x = struct.pack("i", self.i_run)
            curr_mat = event.points[0].mat
            out = struct.pack("iddddii", *self.get_total_jumps_run(run.name), curr_mat)
            self.output_file[x] = out
            self.i_run += 1

    def get_total_jumps(self):
        """Return cumulative jump statistics for every registered run."""
        total_jumps = dict()
        for run in self.runners:
            total_jumps[run] = get_total_jumps_run(run)
        return total_jumps

    def get_total_jumps_run(self, run):
        """Return count, time, displacement, and material for one run."""
        events = self.events[run]
        vec = [0.0, 0.0, 0.0]  # np.zeros_like((3, ))
        i = 0
        jump_time = 0.0
        mat = 999
        for event in events:
            vec = [xyz + val for xyz, val in zip(vec, event[2])]
            i += 1
            jump_time += event[1]
            mat = event[4]
        return (i, jump_time, *vec, mat)


class MultiVerseJumpTracker_v2(BaseOutputProcessor):
    """Buffered DBM variant of the multiverse displacement tracker."""

    def get_defaults(self):
        """Return default buffered-tracker configuration."""
        defa = dict()
        defa["states"] = []
        defa["dim"] = 3
        defa["kr"] = 1.0
        path = os.path.join(os.getcwd(), "mvj.dbm")
        i = 0
        while os.path.isfile(path):
            i += 1
            path = os.path.join(os.getcwd(), f"mvj_{i}.dbm")
        defa["file_path"] = path
        defa["distances"] = dict()
        defa["dist_time"] = dict()
        defa["proc_ids"] = []
        defa["stop_proc_ids"] = []
        defa["dist_prob"] = dict()
        defa["n_write_tasks"] = 32
        defa["n_compacts"] = 128
        defa["write_queue"] = queue.Queue()
        return defa

    def __init__(self, **kwargs):
        """Initialize a buffered multiverse jump tracker."""
        super().__init__()

        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

        if os.path.isfile(self.file_path):
            raise Exception(f"{self.file_path} already exists!")

        self.write_thread = Thread(target=self.write_loop, daemon=True)
        self.write_thread.start()

    def register_run(self, run):
        """Register a run with zero accumulated displacement."""
        super().register_run(run)
        self.distances[run.name] = tuple(0.0 for i in range(self.dim))
        self.dist_time[run.name] = list(self.distances[run.name]) + [None] + [None]

    def write_loop(self):
        """Batch queued contributions and flush them to DBM storage."""
        tasks = []
        while True:
            curr_dist, curr_mat, add_prob = self.write_queue.get()
            tasks.append([curr_dist, curr_mat, add_prob])
            self.write_queue.task_done()
            if len(tasks) >= self.n_write_tasks:
                with dbm.open(self.file_path, "c") as output_file:
                    for task in tasks:
                        curr_dist, curr_mat, add_prob = task
                        self.write_to_file(curr_dist, curr_mat, add_prob, output_file)
                output_file.close(compact=i_compact % self.n_compacts == 0)
                tasks = []

    def write_to_file(self, curr_dist, curr_mat, add_prob, f):
        """Add a contribution to an open DBM file handle."""
        bcurr_dist = struct.pack("dddi", *curr_dist, curr_mat)
        try:
            out = struct.unpack("d", f[bcurr_dist])[0] + add_prob
            assert isinstace(out, float)
            f[bcurr_dist] = struct.pack("d", out)
        except:
            f[bcurr_dist] = struct.pack("d", add_prob)

    def __call__(self, run, data, dt, time):
        """Queue displacement statistics from a processed event."""
        event = data[2]
        if event.proc.proc_idx in self.proc_ids:
            self.distances[run.name] = tuple(
                x + y
                for x, y in zip(
                    self.distances[run.name],
                    event.points[0].distance_vector(event.points[1]),
                )
            )
        if event.proc.proc_idx in self.stop_proc_ids:
            self.dist_time[run.name] = list(self.distances[run.name]) + [time] + [dt]

        curr_mat = event.points[0].mat
        curr_dist = self.distances[run.name]
        add_prob = event.proc.rate.kr / run.st.total()

        try:
            self.dist_prob[(curr_dist, curr_mat)] += add_prob
        except:
            self.dist_prob[(curr_dist, curr_mat)] = add_prob

        self.write_queue.put((curr_dist, curr_mat, add_prob))


class MultiVerseJumpTracker_v3(BaseOutputProcessor):
    """Buffered HDF5 multiverse tracker with periodic checkpoint output."""

    def get_defaults(self):
        """Return default HDF5 tracker configuration."""
        defa = dict()
        defa["states"] = []
        defa["dim"] = 3
        defa["kr"] = 1.0
        path = os.path.join(os.getcwd(), "mvj.dbm")
        i = 0
        while os.path.isfile(path):
            i += 1
            path = os.path.join(os.getcwd(), f"mvj_{i}.dbm")
        defa["file_path"] = path
        defa["distances"] = dict()
        defa["dist_time"] = dict()
        defa["out_dict"] = dict()
        defa["proc_ids"] = []
        defa["stop_proc_ids"] = []
        defa["dist_prob"] = dict()
        defa["n_write_tasks"] = int(2e4)
        defa["write_queue"] = queue.Queue()
        return defa

    def __init__(self, **kwargs):
        """Initialize the tracker and resume existing HDF5 data when available."""
        super().__init__()

        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

        if os.path.isfile(self.file_path):
            if hdfdict is None:
                raise ImportError(
                    "hdfdict is required to resume MultiVerseJumpTracker_v3; "
                    "install the package with the 'hdfdict' dependency"
                )
            warnings.warn(f"{self.file_path} already exists!")
            self.out_dict = dict(hdfdict.load(self.file_path))
            self.out_dict = {
                tuple(eval(kv) for kv in k.split("_")): v
                for k, v in self.out_dict.items()
            }

        self.write_thread = Thread(target=self.write_loop, daemon=True)
        self.write_thread.start()

    def register_run(self, run):
        """Register a run with zero accumulated displacement."""
        super().register_run(run)
        self.distances[run.name] = tuple(0.0 for i in range(self.dim))
        self.dist_time[run.name] = list(self.distances[run.name]) + [None] + [None]

    def write_loop(self):
        """Accumulate queued contributions and periodically checkpoint HDF5 data."""
        tasks = []
        i = 0
        i_f = 0
        while True:
            i += 1
            curr_dist, curr_mat, add_prob = self.write_queue.get()
            try:
                self.out_dict[(curr_dist, curr_mat)] += add_prob
            except:
                self.out_dict[(curr_dist, curr_mat)] = add_prob
            if i >= self.n_write_tasks:
                i = 0
                fpath = self.file_path
                if i_f % 2 == 0:
                    fpath += ".bak"
                try:
                    self.write_to_file(self.out_dict, fpath)
                    i_f += 1
                except Exception as e:
                    warnings.warn(f"Could not write to file {fpath}. Error: {e}")
            self.write_queue.task_done()

    def write_to_file(self, d, f):
        """Write a displacement-probability dictionary to an HDF5 file."""
        if hdfdict is None:
            raise ImportError(
                "hdfdict is required for MultiVerseJumpTracker_v3 output; "
                "install the package with the 'hdfdict' dependency"
            )
        hdfdict.dump(d, h5py.File(f, "w"))

    def __call__(self, run, data, dt, time):
        """Queue displacement statistics from a processed event."""
        event = data[2]
        if event.proc.proc_idx in self.proc_ids:
            self.distances[run.name] = tuple(
                x + y
                for x, y in zip(
                    self.distances[run.name],
                    event.points[0].distance_vector(event.points[1]),
                )
            )
        if event.proc.proc_idx in self.stop_proc_ids:
            self.dist_time[run.name] = list(self.distances[run.name]) + [time] + [dt]

        curr_mat = event.points[0].mat
        curr_dist = self.distances[run.name]
        add_prob = event.proc.rate.kr / run.st.total()

        try:
            self.dist_prob[(curr_dist, curr_mat)] += add_prob
        except:
            self.dist_prob[(curr_dist, curr_mat)] = add_prob

        self.write_queue.put((curr_dist, curr_mat, add_prob))
