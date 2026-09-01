"""Configurable transient TTA experiment runner.

The historical project had one script per combination of kappa model and
crystal lattice. This module keeps the simulation setup in one place and
exposes those two dimensions as settings instead.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

from graphkmc.Crystals import BCC_cell, FCC_cell, SCC_cell, SuperCell
from graphkmc.Lattice import Lattice, OLEDPoint
from graphkmc.PostProcessors import ParticleTracker, RadiativeDecayTracker
from graphkmc.Processes import CustomProcess, ForsterRate, KappaFactor, ProcessTable
from graphkmc.Run import Run

LATTICE_DEFAULTS = {
    "scc": (SCC_cell, 51, 2.5),
    "bcc": (BCC_cell, 41, 1.0),
    "fcc": (FCC_cell, 33, 1.0),
}
KAPPA_MODELS = {"sinkappa", "varkappa"}


def kappa_factor(model: str):
    """Return the kappa factor used by the selected experiment model."""

    model = model.lower()
    if model in {"sinkappa", "conskappa"}:
        return 1.0
    if model == "varkappa":
        return KappaFactor(math.sqrt(3.0 / 2.0))
    raise ValueError(
        f"Unknown kappa model {model!r}; choose sinkappa, conskappa, or varkappa"
    )


def make_lattice(lattice: str, n: int, *, a: float = 1.0, verbose: int = 0) -> Lattice:
    """Create and populate an SCC, BCC, or FCC lattice."""

    lattice = lattice.lower()
    if lattice not in LATTICE_DEFAULTS:
        raise ValueError(f"Unknown lattice {lattice!r}; choose scc, bcc, or fcc")

    cell_type = LATTICE_DEFAULTS[lattice][0]
    scell = SuperCell(n, cell_type(a))
    lat = Lattice(unit_cell=scell.lattice_vectors, verbose=verbose)
    for coords in scell:
        lat.add_point(OLEDPoint(coords=coords, mat=0, verbose=verbose))
    return lat


def _next_output_path(data_dir: Path, prefix: str) -> Path:
    """Return a non-existing output path for a tracker file."""

    index = 0
    while True:
        path = data_dir / f"{prefix}_{index}.txt"
        if not path.exists():
            return path
        index += 1


def run_tta(
    *,
    kappa: str = "varkappa",
    lattice: str = "scc",
    n: int | None = None,
    Rf: float | None = None,
    max_steps: float | None = None,
    max_time: float | None = None,
    verbose: int | None = None,
    seed: int | None = None,
    data_dir: str | os.PathLike[str] | None = None,
) -> Run:
    """Run the transient TTA experiment.

    Parameters ``kappa`` and ``lattice`` select the two former filename
    variants. Both accept ``sinkappa``/``varkappa`` and
    ``scc``/``bcc``/``fcc`` respectively. ``Rf`` overrides the lattice's
    historical default Förster radius when supplied.
    """

    kappa = kappa.lower()
    lattice = lattice.lower()
    if kappa not in KAPPA_MODELS:
        raise ValueError(f"Unknown kappa model {kappa!r}; choose sinkappa or varkappa")
    if lattice not in LATTICE_DEFAULTS:
        raise ValueError(f"Unknown lattice {lattice!r}; choose scc, bcc, or fcc")

    if seed is not None:
        random.seed(seed)

    _, default_n, default_rf = LATTICE_DEFAULTS[lattice]
    n = default_n if n is None else n
    Rf = default_rf if Rf is None else Rf
    kr = 1.0
    a = 1.0
    triplet_density = 0.02
    verbosity = 1000 if verbose is None else verbose
    max_steps = math.inf if max_steps is None else max_steps
    max_time = math.inf if max_time is None else max_time

    output_dir = Path.cwd() / "data" if data_dir is None else Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rcut = max((Rf * Rf) * 9.1 / 9.0, Rf * 3.0, 3.1)
    jobid = os.environ.get("SLURM_JOB_ID", "local")
    prefix = (
        f"tripletracker_{kappa}_{lattice}_n{n}kr{kr:.1f}Rf{Rf:.2f}"
        f"rcut{rcut:.1f}a{a:.1f}td{triplet_density}id{jobid}"
    )
    triptrack_path = _next_output_path(output_dir, prefix)

    krtracker = RadiativeDecayTracker()
    denstracker = ParticleTracker(file_path=str(triptrack_path))
    ops = [krtracker, denstracker]

    lat = make_lattice(lattice, n, a=a, verbose=verbosity)

    lat.initialise_state()
    for point in lat.points:
        point.state = 1 if random.random() <= triplet_density else 0

    pt = ProcessTable()
    kappa_factor_value = kappa_factor(kappa)

    proc = CustomProcess(
        (1, 1),
        (1, 0),
        (0, 0),
        rcut,
        rate=ForsterRate(Rf, kr=kr, kappa=kappa_factor_value),
    )
    pt.add_possible_process(proc)
    denstracker.neg_proc_ids.append(proc.proc_idx)

    proc = CustomProcess((1,), (0,), (0,), 0.4, rate=kr)
    pt.add_possible_process(proc)
    krtracker.proc_ids.append(proc.proc_idx)
    denstracker.states.append(1)
    denstracker.neg_proc_ids.append(proc.proc_idx)

    run = Run(lat, pt, ops, max_steps=max_steps, max_time=max_time, verbose=verbosity)
    run.run()

    while not run.ops[-1].write_queue.empty():
        time.sleep(1.0)

    return run
