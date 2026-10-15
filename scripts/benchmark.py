"""
Runs GASPLAT on a scenario set (see utils/scenario_sets.py) and reports, for
every scenario, the reconstruction metrics of utils.experiment_utils.gas_map_metrics.
Meant to be run after any change to check that nothing got worse. Any
configuration option can be overridden from the command line.

    python scripts/benchmark.py --set tuning
    python scripts/benchmark.py --set held_out --concentration_lr 0.01

Scenarios must be extracted first with scripts/make_scenarios.py.
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
from utils.plot_utils import plot_scenario_grid, set_publication_style
from utils.experiment_utils import yield_scenario_runs, save_experiment_results

COLUMNS = [("nrmse", "NRMSE %", 100, ".2f"), ("peak_ratio", "peak", 1, ".2f"), ("mass_error", "mass %", 100, "+.1f"),
           ("leak", "leak %", 100, ".2f"), ("gaussians", "gauss", 1, "d"), ("poses", "poses", 1, "d"),
           ("beams", "beams", 1, "d"), ("iterations", "iters", 1, "d"), ("time", "time s", 1, ".1f")]


def check_scenarios(dirs: list[str]):
    missing = [d for d in dirs if not os.path.exists(os.path.join(d, "scenario.npz"))]
    if missing:
        sys.exit(f"Missing scenarios: {', '.join(os.path.basename(d) for d in missing)}. "
                 "Extract them with scripts/make_scenarios.py")


def print_table(rows: list[dict]):
    name_w = max(len(r["scenario"]) for r in rows)
    print(f"{'scenario':{name_w}s} " + " ".join(f"{label:>8s}" for _, label, _, _ in COLUMNS))
    for r in rows:
        print(f"{r['scenario']:{name_w}s} " + " ".join(f"{format(r[k] * f, fmt):>8s}" for k, _, f, fmt in COLUMNS))
    nrmse = np.array([r["nrmse"] for r in rows]) * 100
    print(f"NRMSE: mean {nrmse.mean():.2f}%, worst {nrmse.max():.2f}%")


def main():
    parser = ArgumentParser(description="Benchmark GASPLAT on a set of GADEN scenarios")
    parser.add_arguments(Config, dest="cfg")
    parser.add_argument("--set", default="tuning", choices=list(SETS), help="Scenario set to run")
    parser.add_argument("--workers", type=int, default=4, help="Scenarios run in parallel")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False

    set_publication_style()
    dirs = scenario_dirs(args.set)
    check_scenarios(dirs)

    tasks = []
    for d in dirs:
        task_cfg = copy.deepcopy(cfg)
        task_cfg.sim.scenario = d
        tasks.append((os.path.basename(d), task_cfg))

    metrics, preds = {}, {}
    for name, m, pred in tqdm(yield_scenario_runs(tasks, args.workers), total=len(tasks), desc="Scenarios"):
        metrics[name], preds[name] = m, pred

    rows = [metrics[os.path.basename(d)] for d in dirs]
    print_table(rows)

    save_experiment_results(
        {"experiment_name": f"benchmark_{args.set}", "set": args.set, "config": dataclasses.asdict(cfg)},
        {"scenarios": rows,
         "mean_nrmse": float(np.mean([r["nrmse"] for r in rows])),
         "max_nrmse": float(np.max([r["nrmse"] for r in rows]))},
    )
    plot_scenario_grid(dirs, preds, metrics,
                       os.path.join(os.path.dirname(__file__), '..', 'plots', f'benchmark_{args.set}.png'),
                       title=f"Benchmark ({args.set} set)")


if __name__ == "__main__":
    main()
