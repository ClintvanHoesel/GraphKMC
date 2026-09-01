import math
import random

from .SumTree import DynamicSumTree


class SimulationTimer:
    """Mutable simulation-time accumulator."""

    def __init__(self, time=0.0, verbose: int = 0):
        """Initialize the timer at ``time``."""
        self.time = time
        self.verbose = verbose

    def update_time(self, dt):
        """Advance the elapsed time by ``dt``."""
        self.time += dt


class Rate_Sampler:
    """Sample KMC events and waiting times from a dynamic sum tree."""

    def __init__(self, st: DynamicSumTree, seed: int = 1234, verbose: int = 0):
        """Bind the sampler to ``st`` and seed the random number generator."""
        self.st = st
        self.st.rs = self

        random.seed(seed)
        self.verbose = verbose

    def sample(self):
        """Return a weighted event tuple and its exponentially distributed delay."""
        total_rate = self.st.total()
        random_rate = random.uniform(0.0, total_rate)
        event = self.st.get(random_rate)

        random_time = random.uniform(0.0, 1.0)
        dt = (-1.0) * math.log(random_time) / total_rate
        return event, dt
