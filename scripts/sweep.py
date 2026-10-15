"""
One-at-a-time hyperparameter sweep on a scenario set (see utils/scenario_sets.py).
Every listed parameter is varied on its own, the rest keep their current
value, and each parameter gets its own plot and results file.

    # Current value times each factor (default factors: 1/4, 1/2, 1, 2, 4, 8)
    python scripts/sweep.py --params concentration_lr pos_lr scale_lr
    # Explicit values, for parameters without a numeric default or to include 0
    python scripts/sweep.py --params injection_scale --values 0.1 0.25 0.5 1 2
    python scripts/sweep.py --params obstacle_lambda --values 0 0.025 0.05 0.1 0.2 0.4

Parameters are given by name (concentration_lr) or with their group
(train.concentration_lr). Any other configuration option can be overridden as
in main.py, and is kept fixed during the sweep.
"""
import os
import sys
import copy
import dataclasses
import numpy as np
from tqdm import tqdm
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from utils.scenario_sets import SETS, scenario_dirs
from utils.plot_utils import plot_sweep, set_publication_style
from utils.experiment_utils import yield_scenario_runs, save_experiment_results

GROUPS = ("env", "sim", "train", "densify")


def resolve_param(cfg: Config, name: str) -> tuple[str, str]:
    """(group, attribute) of a configuration parameter given as 'attr' or 'group.attr'."""
    if "." in name:
        group, attr = name.split(".", 1)
        if group not in GROUPS or not hasattr(getattr(cfg, group), attr):
            sys.exit(f"Unknown parameter: {name}")
        return group, attr
    matches = [g for g in GROUPS if hasattr(getattr(cfg, g), name)]
    if len(matches) != 1:
        sys.exit(f"Unknown or ambiguous parameter: {name}")
    return matches[0], name


def sweep_values(default, values: list[float] | None, factors: list[float]) -> list:
    if values:
        values = sorted(values)
    elif isinstance(default, (int, float)) and not isinstance(default, bool):
        values = sorted(default * f for f in factors)
    else:
        sys.exit(f"The current value is {default!r}: give the values to try with --values")
    if isinstance(default, int) and not isinstance(default, bool):
        values = sorted({max(1, round(v)) for v in values})
    return values


def summarize(param: str, values: list, runs: dict, scenarios: list[str]):
    print(f"\n{param:>20s} {'NRMSE %':>9s} {'worst %':>9s} {'peak':>7s} {'leak %':>7s} {'gauss':>7s} {'time s':>7s}")
    for v in values:
        rows = [runs[(v, s)] for s in scenarios]
        mean = lambda k: np.mean([r[k] for r in rows])
        print(f"{v:>20g} {100 * mean('nrmse'):9.2f} {100 * max(r['nrmse'] for r in rows):9.2f} "
              f"{mean('peak_ratio'):7.2f} {100 * mean('leak'):7.2f} {mean('gaussians'):7.1f} {mean('time'):7.1f}")


def main():
    parser = ArgumentParser(description="One-at-a-time hyperparameter sweep on GADEN scenarios")
    parser.add_arguments(Config, dest="cfg")
    parser.add_argument("--params", nargs="+", required=True, help="Parameters to sweep, one at a time")
    parser.add_argument("--values", nargs="+", type=float, default=None,
                        help="Values to try (only with a single parameter)")
    parser.add_argument("--factors", nargs="+", type=float, default=[0.25, 0.5, 1, 2, 4, 8],
                        help="Multiples of the current value to try when --values is not given")
    parser.add_argument("--set", default="tuning", choices=list(SETS), help="Scenario set to run")
    parser.add_argument("--workers", type=int, default=4, help="Runs in parallel")
    args = parser.parse_args()
    if args.values and len(args.params) > 1:
        sys.exit("--values can only be used with a single parameter")

    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    set_publication_style()

    dirs = scenario_dirs(args.set)
    missing = [d for d in dirs if not os.path.exists(os.path.join(d, "scenario.npz"))]
    if missing:
        sys.exit(f"Missing scenarios: {', '.join(os.path.basename(d) for d in missing)}. "
                 "Extract them with scripts/make_scenarios.py")
    scenarios = [os.path.basename(d) for d in dirs]

    for name in args.params:
        group, attr = resolve_param(cfg, name)
        default = getattr(getattr(cfg, group), attr)
        values = sweep_values(default, args.values, args.factors)
        # Without a numeric current value, the current configuration is run as a reference
        run_baseline = not isinstance(default, (int, float)) or default not in values

        tasks = []
        for d, scenario in zip(dirs, scenarios):
            for v in values + (["current"] if run_baseline else []):
                task_cfg = copy.deepcopy(cfg)
                task_cfg.sim.scenario = d
                if v != "current":
                    setattr(getattr(task_cfg, group), attr, v)
                tasks.append(((v, scenario), task_cfg))

        runs = {}
        for key, metrics, _ in tqdm(yield_scenario_runs(tasks, args.workers), total=len(tasks),
                                    desc=f"Sweep {group}.{attr}"):
            runs[key] = metrics

        baseline = {s: runs.pop(("current", s)) for s in scenarios} if run_baseline else None
        summarize(f"{group}.{attr}", values, runs, scenarios)
        if baseline is not None:
            nrmse = [baseline[s]["nrmse"] for s in scenarios]
            print(f"{'current (' + str(default) + ')':>20s} {100 * np.mean(nrmse):9.2f} {100 * np.max(nrmse):9.2f}")

        save_experiment_results(
            {"experiment_name": f"sweep_{attr}", "param": f"{group}.{attr}", "values": values,
             "set": args.set, "config": dataclasses.asdict(cfg)},
            {"runs": [{"value": v, **runs[(v, s)]} for v in values for s in scenarios],
             "baseline": baseline},
        )
        plot_sweep(f"{group}.{attr}", values, runs, default,
                   os.path.join(os.path.dirname(__file__), '..', 'plots', f'sweep_{attr}.png'),
                   baseline=baseline)


if __name__ == "__main__":
    main()
