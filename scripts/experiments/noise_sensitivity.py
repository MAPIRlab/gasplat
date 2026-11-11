"""
Reconstruction error of every method as the measurement noise grows, with a fixed
number of measurements.

Every measurement y gets a Gaussian error of standard deviation sigma * y
(cfg.sim.noise_model = "proportional", add_proportional_noise), for each level
sigma of --noise_levels (in %), with the robot at the first --poses vantage poses
of the GADEN scenario (as in num_measurements.py). Run r of a level uses seed r:
the same poses, sweep headings and standard normal draws at every level, so that
levels differ only in the size of the error. Every method reconstructs the same
measurements.

    python scripts/experiments/noise_sensitivity.py --scenario data/scenarios/10x6_empty_room_side_plume_h0.75_t1000 \\
        --poses 4 8 --noise_levels 0 1 2 5 10 20 --repeats 5 --workers 4

With --discrepancy, GASPLAT is also run knowing the noise level (cfg.train.noise_level),
as "Gas Splatting (discrepancy)": its training stops once the data residual reaches the
noise (discrepancy principle). Without noise it is the same as GASPLAT.

Results go to results/noise_sensitivity_<case>_<date>.json; the figure is made by
scripts/figures/noise_figure.py. With --workers > 1 the times are only indicative.
"""
import os
import sys
import copy
import dataclasses
import functools
import numpy as np
from tqdm import tqdm
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from methods.registry import AVAILABLE_METHODS
from utils.scenario_sets import add_case_arguments, cases_label, get_cases
from utils.sim_utils import generate_simulation_data
from utils.experiment_utils import run_methods, save_experiment_results, yield_runs

KEYS = ("nrmse", "peak_ratio", "peak_distance", "mass_error", "time")
DISCREPANCY = "Gas Splatting (discrepancy)"


def run_level(cfg: Config, methods: list[str], discrepancy: bool = False) -> dict:
    batch, environment = generate_simulation_data(cfg)
    metrics, _ = run_methods(batch, environment, cfg, methods)
    if discrepancy:
        known = copy.deepcopy(cfg)
        known.train.noise_level = cfg.sim.noise_level if cfg.sim.noise else None
        metrics[DISCREPANCY] = run_methods(batch, environment, known, ["Gas Splatting"])[0]["Gas Splatting"]
    y = environment.ground_truth.y_true
    rel = float(((batch.measurements - y).norm() / y.norm().clamp_min(1e-12)).item())
    return {"metrics": metrics, "beams": int(batch.measurements.shape[0]), "noise_rms_relative": rel}


def main():
    parser = ArgumentParser(description="Reconstruction error over the measurement noise")
    parser.add_arguments(Config, dest="cfg")
    add_case_arguments(parser, default_set="simple")
    parser.add_argument("--poses", nargs="+", type=int, default=[4, 8], help="Numbers of vantage poses")
    parser.add_argument("--noise_levels", nargs="+", type=float, default=[0, 1, 2, 5, 10, 20],
                        help="Noise levels: std of each measurement's error, in %% of its value")
    parser.add_argument("--methods", nargs="+", default=list(AVAILABLE_METHODS), choices=list(AVAILABLE_METHODS))
    parser.add_argument("--repeats", type=int, default=5, help="Runs per level, seeds 0..N-1")
    parser.add_argument("--workers", type=int, default=1, help="Runs in parallel (times become indicative)")
    parser.add_argument("--discrepancy", action="store_true",
                        help="Also run GASPLAT knowing the noise level (stops at the discrepancy principle)")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet, cfg.train.do_eval, cfg.train.live_vis = True, False, False
    cfg.sim.noise_model = "proportional"

    cases = get_cases(args.set, args.seeds, cfg.sim.scenario)
    if cases[0].scenario is None:
        sys.exit("The noise experiment uses GADEN scenarios (vantage poses)")
    label = cases_label(args.set, args.seeds, cfg.sim.scenario)
    tasks = []
    for c in cases:
        for p in args.poses:
            for level in args.noise_levels:
                for r in range(args.repeats):
                    case_cfg = c.apply(cfg)
                    case_cfg.sim.num_poses, case_cfg.sim.seed = p, r
                    case_cfg.sim.noise, case_cfg.sim.noise_level = level > 0, level / 100
                    tasks.append(((c.name, p, level, r), case_cfg))
    results = {}
    for key, result, _ in tqdm(yield_runs(functools.partial(run_level, methods=args.methods, discrepancy=args.discrepancy), tasks, args.workers,
                                          warmup=True), total=len(tasks), desc="Runs"):
        results[key] = result

    names = [c.name for c in cases]
    methods = args.methods + ([DISCREPANCY] if args.discrepancy else [])
    summary = {}
    for n in names:
        for p in args.poses:
            print(f"\n{n}, {p} poses: NRMSE % (mean ± std over {args.repeats} runs)")
            print(f"  {'noise %':18s}" + "".join(f"{level:>12g}" for level in args.noise_levels))
            for m in methods:
                row = []
                for level in args.noise_levels:
                    runs = [results[(n, p, level, r)]["metrics"][m] for r in range(args.repeats)]
                    stats = {}
                    for k in KEYS:
                        v = [x[k] + (x.get("matrix_time") or 0.0) if k == "time" else x[k] for x in runs]
                        stats[k], stats[k + "_std"] = float(np.mean(v)), float(np.std(v))
                    summary.setdefault(n, {}).setdefault(str(p), {}).setdefault(m, {})[str(level)] = stats
                    row.append(f"{100 * stats['nrmse']:6.2f}±{100 * stats['nrmse_std']:4.2f}")
                print(f"  {m:18s}" + "".join(f"{x:>12s}" for x in row))

    save_experiment_results(
        {"experiment_name": f"noise_sensitivity_{label}", "cases": names, "poses": args.poses,
         "noise_levels": args.noise_levels, "noise_model": "proportional", "methods": methods,
         "repeats": args.repeats, "workers": args.workers, "config": dataclasses.asdict(cfg)},
        {"runs": [{"case": n, "poses": p, "noise_level": level, "seed": r, **results[(n, p, level, r)]}
                  for n in names for p in args.poses for level in args.noise_levels for r in range(args.repeats)],
         "summary": summary})


if __name__ == "__main__":
    main()
