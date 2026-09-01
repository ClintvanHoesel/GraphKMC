"""Implementation for the configurable diffusion-assisted TTA experiment."""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

try:
    from scripts.tta import LATTICE_DEFAULTS, kappa_factor, make_lattice
except ModuleNotFoundError:  # pragma: no cover - exercised by direct execution.
    from tta import LATTICE_DEFAULTS, kappa_factor, make_lattice

from graphkmc.PostProcessors import ParticleTracker, RadiativeDecayTracker
from graphkmc.Processes import CustomProcess, ForsterRate, ProcessTable
from graphkmc.Run import Run


def _next_output_path(data_dir: Path, prefix: str) -> Path:
    """Return a non-existing tracker path for ``prefix``."""

    index = 0
    while True:
        path = data_dir / f"{prefix}_{index}.txt"
        if not path.exists():
            return path
        index += 1


def run_diff_tta(
    *,
    kappa: str = "varkappa",
    lattice: str = "scc",
    n: int | None = None,
    Rf: float | None = None,
    RfDiff: float | None = None,
    max_steps: float | None = None,
    max_time: float | None = None,
    verbose: int | None = None,
    seed: int | None = None,
    data_dir: str | os.PathLike[str] | None = None,
) -> Run:
    """Run diffusion-assisted TTA with configurable physical settings."""

    kappa = kappa.lower()
    lattice = lattice.lower()
    if kappa not in {"conskappa", "sinkappa", "varkappa"}:
        raise ValueError(f"Unknown kappa model {kappa!r}; choose conskappa or varkappa")
    if lattice not in LATTICE_DEFAULTS:
        raise ValueError(f"Unknown lattice {lattice!r}; choose scc, bcc, or fcc")
    if seed is not None:
        random.seed(seed)

    _, default_n, _ = LATTICE_DEFAULTS[lattice]
    n = default_n if n is None else n
    Rf = 3.0 if Rf is None else Rf
    RfDiff = 2.0 if RfDiff is None else RfDiff
    kr = 1.0
    triplet_density = 0.02
    verbosity = 1000 if verbose is None else verbose
    max_steps = math.inf if max_steps is None else max_steps
    max_time = math.inf if max_time is None else max_time

    output_dir = Path.cwd() / "data" if data_dir is None else Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rcut = max((Rf * Rf) * 9.1 / 9.0, Rf * 3.0, 3.1)
    rcut_diff = max((RfDiff * RfDiff) * 9.1 / 9.0, RfDiff * 3.0, 3.1)
    jobid = os.environ.get("SLURM_JOB_ID", "local")
    prefix = (
        f"tripletracker_{kappa}_diff_{lattice}_n{n}kr{kr:.1f}"
        f"Rf{Rf:.2f}RfD{RfDiff:.2f}rcut{rcut:.1f}td{triplet_density}id{jobid}"
    )
    triptrack_path = _next_output_path(output_dir, prefix)

    krtracker = RadiativeDecayTracker()
    denstracker = ParticleTracker(file_path=str(triptrack_path))
    ops = [krtracker, denstracker]
    pt = ProcessTable()
    kappa_value = kappa_factor(kappa)

    proc = CustomProcess(
        (1, 1),
        (1, 0),
        (0, 0),
        rcut,
        rate=ForsterRate(Rf, kr=kr, kappa=kappa_value),
    )
    pt.add_possible_process(proc)
    denstracker.neg_proc_ids.append(proc.proc_idx)

    proc = CustomProcess(
        (1, 0),
        (0, 1),
        (0, 0),
        rcut_diff,
        rate=ForsterRate(RfDiff, kr=kr, kappa=kappa_value),
    )
    pt.add_possible_process(proc)

    proc = CustomProcess((1,), (0,), (0,), 0.4, rate=kr)
    pt.add_possible_process(proc)
    krtracker.proc_ids.append(proc.proc_idx)
    denstracker.states.append(1)
    denstracker.neg_proc_ids.append(proc.proc_idx)

    lat = make_lattice(lattice, n, verbose=verbosity)
    lat.initialise_state()
    for point in lat.points:
        point.state = 1 if random.random() <= triplet_density else 0

    run = Run(lat, pt, ops, max_steps=max_steps, max_time=max_time, verbose=verbosity)
    run.run()
    while not run.ops[-1].write_queue.empty():
        time.sleep(1.0)
    return run
