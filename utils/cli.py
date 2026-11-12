"""
Command-line options shared by the experiment and tool scripts. The cases to run
(--set, --seeds, --iteration) are added by utils.scenario_sets.add_case_arguments.
"""
import sys

from methods.registry import AVAILABLE_METHODS


def add_methods_argument(parser):
    """--methods: the reconstruction methods to run (methods/registry.py), all by default."""
    parser.add_argument("--methods", nargs="+", default=list(AVAILABLE_METHODS), choices=list(AVAILABLE_METHODS),
                        help="Methods to run")


def add_workers_argument(parser, timed: bool = True):
    """--workers: runs in parallel (utils.experiment_utils.yield_runs). Scripts that report
    times run one at a time by default, so that the times are valid; the others default
    to as many as fit (None, see default_workers)."""
    if timed:
        parser.add_argument("--workers", type=int, default=1,
                            help="Runs in parallel. One at a time by default, so that the reported times are "
                                 "valid; with more workers they are only indicative")
    else:
        parser.add_argument("--workers", type=int, default=None,
                            help="Runs in parallel. None: one per GPU, or as many as the CPU cores allow. "
                                 "Times are only indicative")


def reject_options(parser, options: dict[str, str], argv: list[str] | None = None):
    """Stops with an error if the command line sets a configuration field that the script
    sets itself. options maps each field ("sim.seed") to what sets it, for the message.
    A field's flag is recognized with or without its group (--seed, --sim.seed), with
    its value after "=" and abbreviated as argparse accepts it. Call it after
    parser.parse_args(): simple_parsing adds the flags of the configuration while parsing."""
    argv = sys.argv[1:] if argv is None else argv
    flags = list(parser._option_string_actions)
    for arg in argv:
        if arg == "--":
            break
        if not arg.startswith("--"):
            continue
        token = arg.split("=", 1)[0]
        # argparse takes an exact flag, else the only one it abbreviates
        matches = [token] if token in flags else [f for f in flags if f.startswith(token)]
        flag = matches[0] if len(matches) == 1 else token
        for name, reason in options.items():
            if flag in (f"--{name}", f"--{name.split('.')[-1]}"):
                parser.error(f"{token} cannot be set: {reason}")
