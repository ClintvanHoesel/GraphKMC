#!/usr/bin/env python
"""Run the configurable single-diffusion experiment."""

from graphkmc.cli import build_exciton_parser

try:  # Support direct script execution and imports from the repository root.
    from scripts.single_diffusion import run_single_diffusion
except ModuleNotFoundError:  # pragma: no cover - exercised by direct execution.
    from single_diffusion import run_single_diffusion


def main(
    *,
    kappa="varkappa",
    lattice="scc",
    n=None,
    Rf=None,
    n_boxes=1,
    n_reps=500,
    max_steps=None,
    max_time=None,
    verbose=None,
    seed=None,
):
    return run_single_diffusion(
        kappa=kappa,
        lattice=lattice,
        n=n,
        Rf=Rf,
        n_boxes=n_boxes,
        n_reps=n_reps,
        max_steps=max_steps,
        max_time=max_time,
        verbose=verbose,
        seed=seed,
    )


def _build_parser():
    parser = build_exciton_parser(
        "Run single diffusion with configurable kappa model, lattice, and Rf.",
        kappa_choices=("sinkappa", "conskappa", "varkappa"),
    )
    parser.add_argument("--n-boxes", dest="n_boxes", type=int, default=1)
    parser.add_argument("--n-reps", dest="n_reps", type=int, default=500)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(**vars(args))
