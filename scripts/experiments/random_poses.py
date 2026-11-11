"""
Compares the methods over many random placements of the robot's poses.

Each run r of a house draws its poses at random with seed r (cfg.sim.random_poses):
as many as select_vantage_poses needs to cover cfg.sim.coverage of the free space,
at least cfg.sim.min_pose_spacing apart, each sweep starting at a random heading.
Every method reconstructs the same measurements, and GASPLAT is also run without
the obstacle penalty ("Gas Splatting (no penalty)"). Besides the usual metrics, the
free cells of each run are split by how its poses observe them (observation_levels:
no line of sight from any pose, seen along a single direction, covered from two
poses at least cfg.sim.min_crossing_angle apart), and the error and the gas are
reported per region (gas_map_metrics).

    python scripts/experiments/random_poses.py --set all --runs 100 --workers 4
    python scripts/experiments/random_poses.py --set held_out --iteration 200 --runs 100 --workers 4

--iteration runs the VGR houses at an earlier snapshot of their simulation (one every
0.5 s; extract it first with scripts/tools/make_scenarios.py --iteration N).

Runs are appended to results/random_poses_<set>.jsonl as they finish; rerunning
the same command skips the runs already there. The maps of the first --keep_maps
runs of every case are saved in results/random_poses_<set>_maps/. At the end, a
summary over the runs of every case is printed and saved (save_experiment_results).
Regional NRMSEs are averaged over the runs whose region covers at least MIN_AREA
of the free cells.
"""
import os
import sys
import copy
import json
import dataclasses
import functools
import numpy as np
from tqdm import tqdm
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from methods.registry import AVAILABLE_METHODS
from utils.scenario_sets import ITERATION, add_case_arguments, cases_label, get_cases
from utils.sim_utils import generate_simulation_data
from utils.experiment_utils import (REGIONS, default_workers, observation_levels, pose_positions, run_methods,
                                    save_experiment_results, yield_runs)

GASPLAT = "Gas Splatting"
NO_PENALTY = "Gas Splatting (no penalty)"
MIN_AREA = 0.01     # share of the free cells below which a region's NRMSE is not averaged
RESULTS = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def run_case(cfg: Config, methods: list[str], keep_maps: bool) -> dict:
    batch, environment = generate_simulation_data(cfg)
    poses = pose_positions(batch)
    levels = observation_levels(environment.obstacles, cfg.env.cell_size, poses, cfg.sim.max_range,
                                cfg.sim.min_crossing_angle)
    metrics, preds = run_methods(batch, environment, cfg, methods, levels)
    if GASPLAT in methods:
        cfg_no_penalty = copy.deepcopy(cfg)
        cfg_no_penalty.train.obstacle_lambda = 0.0
        m, p = run_methods(batch, environment, cfg_no_penalty, [GASPLAT], levels)
        metrics[NO_PENALTY], preds[NO_PENALTY] = m[GASPLAT], p[GASPLAT]
    result = {"beams": int(batch.measurements.shape[0]), "poses": poses.tolist(), "metrics": metrics}
    if keep_maps:
        result["maps"] = {"gt": environment.ground_truth.gas_map, "obstacles": environment.obstacles,
                          "levels": levels, "poses": poses, **{f"pred/{k}": v for k, v in preds.items()}}
    return result


def summarize(runs: list[dict], names: list[str], methods: list[str]) -> dict:
    """Per case and method: mean, median and std over its runs of every metric, and the
    share of runs where GASPLAT's NRMSE is lower / higher than the method's by > 0.5 points."""
    keys = ["nrmse", "peak_ratio", "peak_distance", "mass_error", "leak", "time", "false_gas_unobserved"]
    keys += [f"{k}_{r}" for r in REGIONS for k in ("nrmse", "gas", "true_gas", "area")]
    keys += [f"{k}_{r}" for r in ("false", "missing") for k in ("nrmse", "area")]
    per_case = {}
    for name in names:
        rs = [r for r in runs if r["case"] == name]
        per_case[name] = {"runs": len(rs)}
        for m in methods:
            out = {}
            for k in keys:
                v = np.array([r["metrics"][m].get(k, np.nan) + ((r["metrics"][m].get("matrix_time") or 0.0)
                                                                  if k == "time" else 0.0) for r in rs], dtype=float)
                if k.startswith("nrmse_"):
                    area = np.array([r["metrics"][m][k.replace("nrmse", "area")] for r in rs])
                    v = np.where(area >= MIN_AREA, v, np.nan)
                if k in ("peak_ratio", "mass_error"):
                    v = np.abs(v - 1) if k == "peak_ratio" else np.abs(v)
                ok = ~np.isnan(v)
                out[k] = {"mean": float(v[ok].mean()) if ok.any() else None,
                          "median": float(np.median(v[ok])) if ok.any() else None,
                          "std": float(v[ok].std()) if ok.any() else None, "n": int(ok.sum())}
            if GASPLAT in methods and m != GASPLAT:
                d = 100 * np.array([r["metrics"][GASPLAT]["nrmse"] - r["metrics"][m]["nrmse"] for r in rs])
                out["gasplat_better"] = float(np.mean(d < -0.5))
                out["gasplat_worse"] = float(np.mean(d > 0.5))
            per_case[name][m] = out
    return per_case


def main():
    parser = ArgumentParser(description="Compare the methods over random pose placements")
    parser.add_arguments(Config, dest="cfg")
    add_case_arguments(parser)
    parser.add_argument("--methods", nargs="+", default=list(AVAILABLE_METHODS), choices=list(AVAILABLE_METHODS))
    parser.add_argument("--runs", type=int, default=100, help="Random pose placements per case, seeds 0..N-1")
    parser.add_argument("--keep_maps", type=int, default=3, help="Save the maps of the first N runs of every case")
    parser.add_argument("--workers", type=int, default=None, help="Runs in parallel (default: as many as fit)")
    parser.add_argument("--iteration", type=int, default=ITERATION,
                        help="Snapshot of the VGR simulations (one every 0.5 s; default: the last)")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet, cfg.train.do_eval, cfg.train.live_vis = True, False, False
    cfg.sim.random_poses = True

    cases = get_cases(args.set, args.seeds, cfg.sim.scenario, iteration=args.iteration)
    if cases[0].scenario is None:
        sys.exit("Random poses apply to GADEN scenarios")
    label = cases_label(args.set, args.seeds, cfg.sim.scenario, args.iteration)
    methods = args.methods + ([NO_PENALTY] if GASPLAT in args.methods else [])
    os.makedirs(RESULTS, exist_ok=True)
    log = os.path.join(RESULTS, f"random_poses_{label}.jsonl")
    maps_dir = os.path.join(RESULTS, f"random_poses_{label}_maps")
    done = set()
    if os.path.exists(log):
        with open(log) as f:
            done = {(r["case"], r["run"]) for r in map(json.loads, f)}

    tasks = []
    for c in cases:
        for r in range(args.runs):
            if (c.name, r) in done:
                continue
            case_cfg = c.apply(cfg)
            case_cfg.sim.seed = r
            tasks.append(((c.name, r), case_cfg, {"keep_maps": r < args.keep_maps}))
    print(f"{len(tasks)} runs to do, {len(done)} already in {log}")
    workers = args.workers or default_workers(cfg)
    for (name, r), result, device in tqdm(yield_runs(functools.partial(run_case, methods=args.methods), tasks,
                                                     workers, warmup=True), total=len(tasks), desc="Runs"):
        maps = result.pop("maps", None)
        if maps is not None:
            os.makedirs(maps_dir, exist_ok=True)
            np.savez_compressed(os.path.join(maps_dir, f"{name}_run{r}.npz"),
                                **{k: np.asarray(v, dtype=np.float32) for k, v in maps.items()})
        with open(log, "a") as f:
            f.write(json.dumps({"case": name, "run": r, "device": device, **result}) + "\n")

    with open(log) as f:
        runs = [json.loads(line) for line in f]
    names = [c.name for c in cases]
    per_case = summarize(runs, names, methods)
    columns = [("nrmse", "all"), ("nrmse_covered", "covered"), ("nrmse_single", "single"),
               ("nrmse_false", "false gas"), ("nrmse_missing", "missing")]
    for name in names:
        x = per_case[name]
        area = x[methods[0]]
        print(f"\n{name} ({x['runs']} runs; unobserved {100 * area['area_unobserved']['mean']:.0f}% of the free "
              f"space, covered {100 * area['area_covered']['mean']:.0f}%; true gas unobserved "
              f"{100 * area['true_gas_unobserved']['mean']:.0f}%)")
        print(f"  {'mean %':26s}" + "".join(f"{c:>10s}" for _, c in columns) + f"{'GS better':>10s}{'worse':>7s}")
        for m in methods:
            row = "".join(f"{100 * x[m][k]['mean']:10.2f}" if x[m][k]["mean"] is not None else f"{'-':>10s}"
                          for k, _ in columns)
            wins = (f"{100 * x[m]['gasplat_better']:9.0f}%{100 * x[m]['gasplat_worse']:6.0f}%"
                    if "gasplat_better" in x[m] else "")
            print(f"  {m:26s}{row}{wins}")
    print(f"\nMean over {len(names)} cases of the per-case means (%)")
    print(f"  {'':26s}" + "".join(f"{c:>10s}" for _, c in columns))
    summary = {}
    for m in methods:
        summary[m] = {k: float(np.mean([per_case[n][m][k]["mean"] for n in names
                                        if per_case[n][m][k]["mean"] is not None])) for k, _ in columns}
        print(f"  {m:26s}" + "".join(f"{100 * summary[m][k]:10.2f}" for k, _ in columns))

    save_experiment_results(
        {"experiment_name": f"random_poses_{label}", "cases": names, "methods": methods, "runs": args.runs,
         "log": os.path.abspath(log), "min_area": MIN_AREA, "iteration": args.iteration, "config": dataclasses.asdict(cfg)},
        {"per_case": per_case, "summary": summary})


if __name__ == "__main__":
    main()
