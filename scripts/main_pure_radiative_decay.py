#!/usr/bin/env python
"""Run the Main Pure Radiative Decay experiment.

This legacy experiment is intentionally kept as a script, but all work
is performed by :func:`main` so importing the file has no side effects.
"""

import math
import random

import numpy as np

from graphkmc.cli import build_experiment_parser
from graphkmc.Lattice import Lattice, Point
from graphkmc.Processes import CustomProcess, ProcessTable
from graphkmc.Run import ParticleTracker, RadiativeDecayTracker, Run


def main(*, n=None, max_steps=None, max_time=None, verbose=None, seed=None):
    if seed is not None:
        random.seed(seed)
    max_steps = math.inf if max_steps is None else max_steps
    max_time = math.inf if max_time is None else max_time

    ops = []
    krtracker = RadiativeDecayTracker()
    ops.append(krtracker)
    denstracker = ParticleTracker()
    ops.append(denstracker)

    n = 10 if n is None else n
    triplet_density = 1.0

    unit_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * n
    lat = Lattice(unit_cell=unit_cell)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                lat.add_point(Point(coords=np.array([i, j, k]), mat=0))

    lat.initialise_state()

    for point in lat.points:
        point.state = 0
        if random.random() <= triplet_density:
            point.state = 1

    pt = ProcessTable(lat)

    inits = tuple([1])
    fins = tuple([0])
    mats = tuple([0])
    rcut = 0.4
    proc = CustomProcess(inits, fins, mats, rcut, rate=1.0)
    pt.add_possible_process(proc)
    krtracker.proc_ids.append(proc.proc_idx)
    denstracker.states.append(1)
    denstracker.neg_proc_ids.append(proc.proc_idx)

    run = Run(
        lat, pt, ops, max_steps=max_steps, max_time=max_time, verbose=verbose, seed=seed
    )
    run.run()

    return run


def _build_parser():

    return build_experiment_parser("Run the Main Pure Radiative Decay experiment.")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(**vars(args))
