#!/usr/bin/env python
"""Run the configurable diffusion-assisted TTA experiment."""

from graphkmc.cli import build_exciton_parser

try:  # Support direct script execution and imports from the repository root.
    from scripts.diff_tta import run_diff_tta
except ModuleNotFoundError:  # pragma: no cover - exercised by direct execution.
    from diff_tta import run_diff_tta


def main(
    *,
    kappa="varkappa",
    lattice="scc",
    n=None,
    Rf=None,
    RfDiff=None,
    max_steps=None,
    max_time=None,
    verbose=None,
    seed=None,
):
    """Run diffusion-assisted TTA for the selected settings."""

    return run_diff_tta(
        kappa=kappa,
        lattice=lattice,
        n=n,
        Rf=Rf,
        RfDiff=RfDiff,
        max_steps=max_steps,
        max_time=max_time,
        verbose=verbose,
        seed=seed,
    )


def _build_parser():
    parser = build_exciton_parser(
        "Run diffusion-assisted TTA with configurable kappa model and lattice.",
        kappa_choices=("sinkappa", "conskappa", "varkappa"),
    )
    parser.add_argument(
        "--RfDiff",
        "--rf-diff",
        "--RFDiff",
        dest="RfDiff",
        type=float,
        default=None,
        help="Förster radius for the diffusion channel (default: 2.0).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(**vars(args))
