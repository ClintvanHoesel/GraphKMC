#!/usr/bin/env python
"""Run the configurable transient TTA experiment."""

from graphkmc.cli import build_exciton_parser

try:  # Support both ``python scripts/main_TTA.py`` and module-style imports.
    from scripts.tta import run_tta
except ModuleNotFoundError:  # pragma: no cover - exercised by direct execution.
    from tta import run_tta


def main(
    *,
    kappa="varkappa",
    lattice="scc",
    Rf=None,
    n=None,
    max_steps=None,
    max_time=None,
    verbose=None,
    seed=None,
):
    """Run TTA for the selected kappa model and crystal lattice."""

    return run_tta(
        kappa=kappa,
        lattice=lattice,
        Rf=Rf,
        n=n,
        max_steps=max_steps,
        max_time=max_time,
        verbose=verbose,
        seed=seed,
    )


def _build_parser():
    return build_exciton_parser(
        "Run transient TTA with configurable kappa model and lattice."
    )


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(**vars(args))
