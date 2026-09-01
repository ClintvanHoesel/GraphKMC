"""Command-line entry points for GraphKMC."""

import argparse

from .Preset_Runners import SteadyStateRunnerTTA


def _bool_string(value: str) -> bool:
    """Parse a case-insensitive ``true`` or ``false`` command-line value."""
    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_experiment_parser(description: str) -> argparse.ArgumentParser:
    """Build the parser shared by parameterised experiment scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Linear lattice size (default: experiment value).",
    )
    parser.add_argument(
        "--max-steps",
        dest="max_steps",
        type=float,
        default=None,
        help="Maximum KMC steps.",
    )
    parser.add_argument(
        "--max-time",
        dest="max_time",
        type=float,
        default=None,
        help="Maximum simulation time.",
    )
    parser.add_argument("--verbose", type=int, default=None, help="Progress verbosity.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser


def build_exciton_parser(
    description: str,
    *,
    kappa_choices=("sinkappa", "varkappa"),
    default_kappa="varkappa",
) -> argparse.ArgumentParser:
    """Build the common parser for the configurable transient exciton runners."""

    parser = build_experiment_parser(description)
    parser.add_argument(
        "--kappa",
        "--kappa-type",
        dest="kappa",
        choices=kappa_choices,
        default=default_kappa,
        help=f"Kappa model (default: {default_kappa}).",
    )
    parser.add_argument(
        "--lattice",
        "--lattice-type",
        "--latticetype",
        dest="lattice",
        choices=("scc", "bcc", "fcc"),
        default="scc",
        help="Crystal lattice (default: scc).",
    )
    parser.add_argument(
        "--Rf",
        "--rf",
        "--RF",
        dest="Rf",
        type=float,
        default=None,
        help="Förster radius (default: selected lattice value).",
    )
    return parser


def main(argv=None):
    """Run the installed steady-state TTA command-line interface."""
    parser = argparse.ArgumentParser(description="Steady-state GraphKMC TTA runner")
    parser.add_argument("--Rf", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--max_time", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=int(1e7))
    parser.add_argument("--varKappa", type=_bool_string, default=True)
    parser.add_argument("--latticetype", choices=("scc", "bcc", "fcc"), default="scc")
    args = parser.parse_args(argv)
    runner = SteadyStateRunnerTTA(**vars(args))
    runner.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
