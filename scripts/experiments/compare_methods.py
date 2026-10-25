"""
Compares GASPLAT with the grid-based tomography baselines (methods/registry.py):
every method reconstructs the same measurements of each case, and their
reconstruction error and time are reported, per case and on average.

    python scripts/experiments/compare_methods.py --set held_out --repeats 5
    python scripts/experiments/compare_methods.py --seeds 0 1 2 --methods SART "Gas Splatting"

--repeats N runs every GADEN case N times with seeds 0..N-1, which only change the
starting heading of each sweep; every run is saved, and a case is summarized by the
mean over its runs. Averages over cases weigh every case once, and use absolute
values for the signed errors (peak ratio - 1, mass error), so that an over- and an
underestimate do not cancel. Times of the grid methods include the system matrix.

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
from utils.scenario_sets import add_case_arguments, cases_label, get_cases
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
    parser.add_argument("--repeats", type=int, default=1, help="Runs per GADEN case, with seeds 0..N-1 (sweep headings)")
    parser.add_argument("--workers", type=int, default=1, help="Cases run in parallel (times become indicative)")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    set_publication_style()

    cases = get_cases(args.set, args.seeds, cfg.sim.scenario)
    label = cases_label(args.set, args.seeds, cfg.sim.scenario)
    procedural = cases[0].scenario is None
    if procedural and args.repeats > 1:
        sys.exit("--repeats applies to GADEN scenarios; procedural maps vary with --seeds")
    repeats = list(range(args.repeats)) if args.repeats > 1 else [cfg.sim.seed]
    tasks = []
    for c in cases:
        for r in repeats:
            case_cfg = c.apply(cfg)
            if not procedural:
                case_cfg.sim.seed = r
            tasks.append(((c.name, r), case_cfg))
    results = {}
    for key, result, device in tqdm(yield_runs(functools.partial(compare_case, methods=args.methods), tasks,
                                               args.workers, warmup=True), total=len(tasks), desc="Runs"):
        results[key] = {**result, "device": device}

    def total_time(x):
        return x["time"] + (x.get("matrix_time") or 0.0)

    # Per case: mean over its runs
    names = [c.name for c in cases]
    keys = ("nrmse", "peak_error", "peak_distance", "mass_error", "time")
    per_case = {}
    for name in names:
        runs = [results[(name, r)] for r in repeats]
        per_case[name] = {m: {"nrmse": float(np.mean([x["metrics"][m]["nrmse"] for x in runs])),
                              "peak_error": float(np.mean([abs(x["metrics"][m]["peak_ratio"] - 1) for x in runs])),
                              "peak_distance": float(np.mean([x["metrics"][m]["peak_distance"] for x in runs])),
                              "mass_error": float(np.mean([abs(x["metrics"][m]["mass_error"]) for x in runs])),
                              "time": float(np.mean([total_time(x["metrics"][m]) for x in runs]))}
                          for m in args.methods}
        print(f"\n{name} ({runs[0]['beams']} beams, {len(runs)} runs)")
        print(f"  {'method':18s} {'NRMSE %':>8s} {'peak err %':>10s} {'peak dist m':>11s} {'mass err %':>10s} {'time s':>8s}")
        for m in args.methods:
            x = per_case[name][m]
            print(f"  {m:18s} {100 * x['nrmse']:8.2f} {100 * x['peak_error']:10.1f} {x['peak_distance']:11.2f} "
                  f"{100 * x['mass_error']:10.1f} {x['time']:8.2f}")

    # Over cases: mean and standard deviation, every case weighing once
    print(f"\nMean (std) over {len(names)} cases ({label})")
    print(f"  {'method':18s} {'NRMSE %':>14s} {'peak err %':>14s} {'peak dist m':>14s} {'mass err %':>14s} {'time s':>12s}")
    summary = {}
    for m in args.methods:
        summary[m] = {}
        for k in keys:
            v = [per_case[n][m][k] for n in names]
            summary[m][k] = float(np.mean(v))
            summary[m][k + "_std"] = float(np.std(v))
        f = lambda k, scale=100: f"{scale * summary[m][k]:7.2f} ({scale * summary[m][k + '_std']:4.2f})"
        print(f"  {m:18s} {f('nrmse'):>14s} {f('peak_error'):>14s} {f('peak_distance', 1):>14s} "
              f"{f('mass_error'):>14s} {summary[m]['time']:7.2f} ({summary[m]['time_std']:.2f})")
    best = {n: min(args.methods, key=lambda m: per_case[n][m]["nrmse"]) for n in names}
    print("  Best NRMSE per case:", {m: sum(b == m for b in best.values()) for m in args.methods})

    save_experiment_results(
        {"experiment_name": f"compare_methods_{label}", "cases": names, "methods": args.methods,
         "repeats": repeats, "workers": args.workers, "config": dataclasses.asdict(cfg)},
        {"runs": [{"case": n, "seed": r, "beams": results[(n, r)]["beams"], "device": results[(n, r)]["device"],
                   "metrics": results[(n, r)]["metrics"]} for n in names for r in repeats],
         "per_case": per_case, "summary": summary},
    )
    first = repeats[0]
    maps = {n: results[(n, first)]["maps"] for n in names}
    plot_reconstruction_grid(
        names, maps, {m: {n: maps[n]["preds"][m] for n in names} for m in args.methods},
        {(m, n): f"NRMSE {100 * results[(n, first)]['metrics'][m]['nrmse']:.1f}%  "
                 f"{total_time(results[(n, first)]['metrics'][m]):.2f} s"
         for m in args.methods for n in names},
        os.path.join(os.path.dirname(__file__), '..', '..', 'plots', f'compare_methods_{label}.png'),
        title=f"Method comparison ({label}, seed {first})")


if __name__ == "__main__":
    main()
