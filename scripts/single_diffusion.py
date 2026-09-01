"""Implementation for the configurable single-diffusion experiment."""

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

from graphkmc.PostProcessors import (
    JumpTracker,
    MultiVerseJumpTracker,
    RadiativeDecayTracker,
)
from graphkmc.Processes import CustomProcess, ForsterRate, ProcessTable
from graphkmc.Run import Run


def _next_output_paths(data_dir: Path, prefix: str) -> tuple[Path, Path]:
    index = 0
    while True:
        pos_path = data_dir / f"jump_vector_{prefix}_{index}.txt"
        mvj_path = data_dir / f"mvj_vector_{prefix}_{index}.shelve"
        if not pos_path.exists() and not mvj_path.exists():
            return pos_path, mvj_path
        index += 1


def run_single_diffusion(
    *,
    kappa: str = "varkappa",
    lattice: str = "scc",
    n: int | None = None,
    Rf: float | None = None,
    n_boxes: int = 1,
    n_reps: int = 500,
    max_steps: float | None = None,
    max_time: float | None = None,
    verbose: int | None = None,
    seed: int | None = None,
    data_dir: str | os.PathLike[str] | None = None,
) -> Run:
    """Run single-particle diffusion with configurable model settings."""

    kappa = kappa.lower()
    lattice = lattice.lower()
    if kappa not in {"conskappa", "sinkappa", "varkappa"}:
        raise ValueError(f"Unknown kappa model {kappa!r}; choose conskappa or varkappa")
    if lattice not in LATTICE_DEFAULTS:
        raise ValueError(f"Unknown lattice {lattice!r}; choose scc, bcc, or fcc")
    if n_boxes < 1 or n_reps < 1:
        raise ValueError("n_boxes and n_reps must be positive")
    if seed is not None:
        random.seed(seed)

    _, default_n, _ = LATTICE_DEFAULTS[lattice]
    n = default_n if n is None else n
    Rf = 3.0 if Rf is None else Rf
    kr = 1.0
    verbosity = 0 if verbose is None else verbose
    max_steps = math.inf if max_steps is None else max_steps
    max_time = math.inf if max_time is None else max_time

    output_dir = Path.cwd() / "data" if data_dir is None else Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rcut = max((Rf * Rf) * 9.1 / 9.0, Rf * 3.0, 3.1)
    jobid = os.environ.get("SLURM_JOB_ID", "local")
    prefix = (
        f"{kappa}_{lattice}_n{n}kr{kr:.1f}Rf{Rf:.2f}rcut{rcut:.1f}"
        f"nboxes{n_boxes}nreps{n_reps}id{jobid}"
    )
    pos_path, mvj_path = _next_output_paths(output_dir, prefix)

    krtracker = RadiativeDecayTracker()
    jumptracker = JumpTracker(file_path=str(pos_path))
    mvjtracker = MultiVerseJumpTracker(file_path=str(mvj_path))
    ops = [krtracker, jumptracker, mvjtracker]

    pt = ProcessTable()
    proc = CustomProcess(
        (1, 0),
        (0, 1),
        (0, 0),
        rcut,
        rate=ForsterRate(Rf, kr=kr, kappa=kappa_factor(kappa)),
    )
    pt.add_possible_process(proc)
    jumptracker.proc_ids.append(proc.proc_idx)
    mvjtracker.proc_ids.append(proc.proc_idx)

    proc = CustomProcess((1,), (0,), (0,), 0.4, rate=kr)
    pt.add_possible_process(proc)
    krtracker.proc_ids.append(proc.proc_idx)
    jumptracker.stop_proc_ids.append(proc.proc_idx)
    mvjtracker.stop_proc_ids.append(proc.proc_idx)

    run = None
    for ibox in range(n_boxes):
        lat = make_lattice(lattice, n, verbose=verbosity)
        lat.initialise_state()
        for point in lat.points:
            point.state = 0
        lat.points[random.randrange(len(lat.points))].state = 1

        run = Run(
            lat,
            pt,
            ops,
            max_steps=max_steps,
            max_time=max_time,
            name=f"{ibox}_0",
            verbose=verbosity,
        )
        run.run()
        _write_jump_row(pos_path, jumptracker, run, ibox, 0)

        for irun in range(1, n_reps):
            lat.points[random.randrange(len(lat.points))].process_state_change(1)
            run.reset(new_name=f"{ibox}_{irun}", reinit_events=False)
            run.run()
            _write_jump_row(pos_path, jumptracker, run, ibox, irun)

    if run is None:  # pragma: no cover
        raise ValueError("n_boxes must be greater than zero")
    while not run.ops[-1].write_queue.empty():
        time.sleep(1.0)
    return run


def _write_jump_row(path: Path, tracker, run: Run, ibox: int, irun: int) -> None:
    values = list(tracker.get_total_jumps_run(run.name)) + [ibox, irun]
    with path.open("a") as output:
        output.write(",".join(str(value) for value in values) + "\r\n")
