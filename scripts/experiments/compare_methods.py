"""
Compares GASPLAT with the grid-based tomography baselines (methods/registry.py):
every method reconstructs the same measurements of each case, and their
reconstruction error and time are reported, per case and on average.

    python scripts/experiments/compare_methods.py --set tuning
    python scripts/experiments/compare_methods.py --seeds 0 1 2 --methods SART "Gas Splatting"

The system matrix shared by the grid methods is timed apart from their solves.
Cases run one at a time by default, and every process makes a throwaway run
first, so that times are comparable; with --workers > 1 they are only indicative.
"""
import os
import sys
import dataclasses
import functools
import numpy as np
from tqdm import tqdm
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from methods.registry import AVAILABLE_METHODS
from utils.scenario_sets import add_case_arguments, get_cases
from utils.sim_utils import generate_simulation_data
from utils.plot_utils import plot_reconstruction_grid, set_publication_style
from utils.experiment_utils import evaluation_maps, run_methods, save_experiment_results, yield_runs


def compare_case(cfg: Config, methods: list[str]) -> dict:
    batch, environment = generate_simulation_data(cfg)
    metrics, preds = run_methods(batch, environment, cfg, methods)
    return {"metrics": metrics, "maps": {**evaluation_maps(environment, cfg), "preds": preds},
            "beams": int(batch.measurements.shape[0])}


def main():
    parser = ArgumentParser(description="Compare GASPLAT with the tomography baselines")
    parser.add_arguments(Config, dest="cfg")
    add_case_arguments(parser)
    parser.add_argument("--methods", nargs="+", default=list(AVAILABLE_METHODS), choices=list(AVAILABLE_METHODS))
    parser.add_argument("--workers", type=int, default=1, help="Cases run in parallel (times become indicative)")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    set_publication_style()

    cases = get_cases(args.set, args.seeds)
    label = "procedural" if args.seeds else args.set
    tasks = [(c.name, c.apply(cfg)) for c in cases]
    results = {}
    for name, result, device in tqdm(yield_runs(functools.partial(compare_case, methods=args.methods), tasks,
                                                args.workers, warmup=True), total=len(tasks), desc="Cases"):
        results[name] = {**result, "device": device}

    names = [c.name for c in cases]
    for name in names:
        r = results[name]
        print(f"\n{name} ({r['beams']} beams, {r['device']})")
        print(f"  {'method':18s} {'NRMSE %':>8s} {'SSIM':>6s} {'peak':>6s} {'mass %':>7s} {'leak %':>7s} {'time s':>8s}")
        for m in args.methods:
            x = r["metrics"][m]
            print(f"  {m:18s} {100 * x['nrmse']:8.2f} {x['ssim']:6.3f} {x['peak_ratio']:6.2f} "
                  f"{100 * x['mass_error']:+7.1f} {100 * x['leak']:7.2f} {x['time']:8.2f}")
    matrix_times = [r["metrics"][m]["matrix_time"] for r in results.values() for m in args.methods
                    if r["metrics"][m]["matrix_time"] is not None]

    print(f"\nAverage over {len(names)} cases ({label})")
    print(f"  {'method':18s} {'NRMSE %':>8s} {'worst %':>8s} {'SSIM':>6s} {'time s':>8s}")
    summary = {}
    for m in args.methods:
        rows = [results[n]["metrics"][m] for n in names]
        summary[m] = {k: float(np.mean([x[k] for x in rows])) for k in ("nrmse", "ssim", "peak_ratio", "mass_error", "leak", "time")}
        summary[m]["max_nrmse"] = float(max(x["nrmse"] for x in rows))
        print(f"  {m:18s} {100 * summary[m]['nrmse']:8.2f} {100 * summary[m]['max_nrmse']:8.2f} "
              f"{summary[m]['ssim']:6.3f} {summary[m]['time']:8.2f}")
    if matrix_times:
        print(f"  System matrix of the grid methods: {np.mean(matrix_times):.3f} s on average (not included above)")

    save_experiment_results(
        {"experiment_name": f"compare_methods_{label}", "cases": names, "methods": args.methods,
         "workers": args.workers, "config": dataclasses.asdict(cfg)},
        {"cases": {n: {"beams": results[n]["beams"], "device": results[n]["device"], "metrics": results[n]["metrics"]}
                   for n in names},
         "summary": summary},
    )
    maps = {n: results[n]["maps"] for n in names}
    plot_reconstruction_grid(
        names, maps, {m: {n: maps[n]["preds"][m] for n in names} for m in args.methods},
        {(m, n): f"NRMSE {100 * results[n]['metrics'][m]['nrmse']:.1f}%  {results[n]['metrics'][m]['time']:.2f} s"
         for m in args.methods for n in names},
        os.path.join(os.path.dirname(__file__), '..', '..', 'plots', f'compare_methods_{label}.png'),
        title=f"Method comparison ({label})")


if __name__ == "__main__":
    main()
