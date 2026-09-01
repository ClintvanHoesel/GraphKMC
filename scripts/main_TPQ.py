#!/usr/bin/env python
"""Run the configurable TPQ experiment."""

from graphkmc.cli import build_exciton_parser

try:  # Support direct script execution and imports from the repository root.
    from scripts.tpq import run_tpq
except ModuleNotFoundError:  # pragma: no cover - exercised by direct execution.
    from tpq import run_tpq


def main(
    *,
    kappa="conskappa",
    lattice="scc",
    n=None,
    Rf=None,
    max_steps=None,
    max_time=None,
    verbose=None,
    seed=None,
):
    return run_tpq(
        kappa=kappa,
        lattice=lattice,
        n=n,
        Rf=Rf,
        max_steps=max_steps,
        max_time=max_time,
        verbose=verbose,
        seed=seed,
    )


def _build_parser():
    return build_exciton_parser(
        "Run TPQ with configurable kappa model, lattice, and Förster radius.",
        kappa_choices=("sinkappa", "varkappa"),
        default_kappa="sinkappa",
    )


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(**vars(args))
