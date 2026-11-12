"""
Runs GASPLAT on a scenario set (see utils/scenario_sets.py) and reports, for
every scenario, the reconstruction metrics of utils.metrics.gas_map_metrics.
Meant to be run after any change to check that nothing got worse. Any
configuration option can be overridden from the command line, which also makes
ablations a single run each (e.g. --original_dens for the 3DGS densification).

    python scripts/tools/benchmark.py --set tuning
    python scripts/tools/benchmark.py --set held_out --concentration_lr 0.01
    python scripts/tools/benchmark.py --seeds 0 1 2          # procedural maps

GADEN scenarios must be extracted first with scripts/tools/make_scenarios.py.
"""
import os
import sys
import copy
import dataclasses
import numpy as np
from tqdm import tqdm
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from utils.paths import PLOTS_DIR
from utils.cli import add_workers_argument
from utils.scenario_sets import add_case_arguments, cases_label, get_cases
from utils.plot_utils import plot_reconstruction_grid
from utils.figure_style import apply_plot_style
from utils.experiment_utils import batch_config, default_workers, yield_scenario_runs, save_experiment_results

COLUMNS = [("nrmse", "NRMSE %", 100, ".2f"), ("peak_ratio", "peak", 1, ".2f"), ("mass_error", "mass %", 100, "+.1f"),
           ("leak", "leak %", 100, ".2f"), ("gaussians", "gauss", 1, "d"), ("poses", "poses", 1, "d"),
           ("beams", "beams", 1, "d"), ("iterations", "iters", 1, "d"), ("time", "time s", 1, ".1f")]


def print_table(rows: list[dict]):
    name_w = max(len(r["scenario"]) for r in rows)
    print(f"{'scenario':{name_w}s} " + " ".join(f"{label:>8s}" for _, label, _, _ in COLUMNS))
    for r in rows:
        print(f"{r['scenario']:{name_w}s} " + " ".join(
            f"{format(r[k] * f, fmt) if r[k] is not None else '-':>8s}" for k, _, f, fmt in COLUMNS))
    nrmse = np.array([r["nrmse"] for r in rows]) * 100
    print(f"NRMSE: mean {nrmse.mean():.2f}%, worst {nrmse.max():.2f}%")


def main():
    parser = ArgumentParser(description="Benchmark GASPLAT on a set of GADEN scenarios")
    parser.add_arguments(Config, dest="cfg")
    add_case_arguments(parser)
    add_workers_argument(parser, timed=False)
    args = parser.parse_args()
    cfg: Config = batch_config(args.cfg)

    apply_plot_style()
    cases = get_cases(args.set, args.seeds, cfg.sim.scenario, iteration=args.iteration)
    workers = args.workers or default_workers(cfg)
    label = cases_label(args.set, args.seeds, cfg.sim.scenario, args.iteration)
    tasks = [(c.name, c.apply(cfg)) for c in cases]

    metrics, maps = {}, {}
    for name, m, case_maps in tqdm(yield_scenario_runs(tasks, workers), total=len(tasks), desc="Scenarios"):
        metrics[name], maps[name] = m, case_maps

    names = [c.name for c in cases]
    rows = [metrics[n] for n in names]
    print_table(rows)

    save_experiment_results(
        {"experiment_name": f"benchmark_{label}", "cases": names, "config": dataclasses.asdict(cfg)},
        {"scenarios": rows,
         "mean_nrmse": float(np.mean([r["nrmse"] for r in rows])),
         "max_nrmse": float(np.max([r["nrmse"] for r in rows]))},
    )
    plot_reconstruction_grid(
        names, maps, {"GASPLAT": {n: maps[n]["pred"] for n in names}},
        {("GASPLAT", n): f"NRMSE {100 * metrics[n]['nrmse']:.1f}%\npeak x{metrics[n]['peak_ratio']:.2f}" for n in names},
        os.path.join(PLOTS_DIR, f'benchmark_{label}.png'),
        title=f"Benchmark ({label})")


if __name__ == "__main__":
    main()
