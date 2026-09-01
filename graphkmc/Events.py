import warnings

from tqdm import tqdm

from .Lattice import Lattice, Point
from .Processes import CustomProcess, ProcessTable
from .SumTree import DynamicSumTree


class Event:
    """A currently possible process execution and its rate contribution."""

    def __init__(
        self, points: tuple, proc: CustomProcess, st: DynamicSumTree, **kwargs
    ):
        """Create an event connecting ``points`` through ``proc``."""
        self.points = points
        self.proc = proc
        self.st = st

        self.verbose = 0

        self.callbacks = self.proc.callbacks

        self.registered_at_points = False
        self.registered_at_sumtree = False

        self.__dict__.update(kwargs)

    def register(self):
        """Register the event with its points and the rate sum tree."""
        self.register_points()
        self.register_sumtree()

    def register_points(self):
        """Attach this event to every participating point once."""
        if not self.registered_at_points:
            for point in self.points:
                point.events.append(self)
            self.registered_at_points = True

    def register_sumtree(self):
        """Insert this event's rate and payload into the dynamic sum tree."""
        if not self.registered_at_sumtree:
            idx = self.st.add(self.rate, self)
            self.idx = idx
            self.registered_at_sumtree = True

    def __repr__(self):
        """Return a compact representation without cyclic references."""
        string = "Event("
        for key, val in self.__dict__.items():
            if key not in ["st", "lat", "et", "points"]:
                string += f"{key}={val},"
            else:
                string += f"{key}=(...),"
        string += ")"
        return string

    def remove(self):
        """Remove the event from points and the rate sum tree."""
        if self.registered_at_points:
            for point in self.points:
                point.events.remove(self)

        if self.registered_at_sumtree:
            self.st.delete(self.idx)

    def process(self):
        """Apply the process state transition represented by this event."""
        return self.proc.process(**self.__dict__)

    @property
    def rate(self):
        """Return the current process rate for the event's points."""
        out = self.proc(**self.__dict__)
        return out


class EventTable:
    """Create, own, and process the currently available KMC events."""

    def get_defaults(self):
        """Return default event-table configuration values."""
        defa = dict()
        defa["verbose"] = 0
        return defa

    def __init__(self, lat: Lattice, st: DynamicSumTree, **kwargs):
        """Bind an event table to a lattice and dynamic sum tree."""
        self.lat = lat
        self.lat.et = self

        self.st = st
        self.st.et = self

        self.__dict__.update(kwargs)
        self.__dict__ = self.get_defaults() | self.__dict__

    def n_events(self):
        """Return the number of stored events."""
        return st.data.n_entries

    def add(self, event: Event):
        """Register ``event`` in the table's point and rate structures."""
        return event.register()

    def __len__(self):
        """Return the number of active event entries."""
        return len(self.st.taken_entries)

    @property
    def events(self):
        """Return an immutable snapshot of active events."""
        return tuple(self.st.data[idx] for idx in self.st.taken_entries)

    def get_single_particle_events(self, point: Point):
        """Create eligible one-point events for ``point``."""
        for proc in point.single_procs:
            if (point.state,) == proc.init_state:
                event = Event((point,), proc, self.st)
                self.add(event)

    def add_single_particle_events(self, point: Point):
        """Add eligible one-point events for ``point``."""
        for proc in point.single_procs:
            if (point.state,) == proc.init_state:
                event = Event((point,), proc, self.st)
                self.add(event)

    def add_active_particle_events(self, point: Point):
        """Add two-point events for processes active at ``point``."""
        for proc in point.active_procs:
            for neighbour in point.proc_neighbours[proc]:
                if (point.state, neighbour.state) == proc.init_state:
                    event = Event((point, neighbour), proc, self.st)
                    self.add(event)

    def add_undergone_particle_events(self, point: Point):
        """Add two-point events where ``point`` is the second participant."""
        for proc in point.undergone_procs:
            for neighbour in point.proc_neighbours[proc]:
                if (neighbour.state, point.state) == proc.init_state:
                    event = Event((neighbour, point), proc, self.st)
                    self.add(event)

    def initialise_events(self):
        """Populate the table with events allowed by the current lattice state."""
        if self.verbose > 3:
            itt = tqdm(self.lat.points, desc="Events")
        else:
            itt = self.lat.points
        for point in itt:
            point.et = self
            warnings.warn("Checking of state might still go wrong")
            self.get_single_particle_events(point)

            self.add_active_particle_events(point)

    def process_event(self, event: Event):
        """Apply a selected event to the lattice."""
        event.process()

    def __repr__(self):
        """Return a compact representation of the event table."""
        string = "EventTable("
        for key, val in self.__dict__.items():
            if key not in ["lat"]:
                string += f"{key}={val},"
            else:
                string += f"{key}=(...),"
        string += ")"
        return string
