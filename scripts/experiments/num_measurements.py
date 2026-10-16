"""
Reconstruction error as measurements are added, for every method.

On GADEN scenarios the measurements grow pose by pose, in the order the
coverage selection picks the poses (each one adds the most newly covered
free space), as a robot would visit them. Procedural maps have no poses, so
their beams are added in a fixed random order instead. At every step each
method reconstructs from the measurements gathered so far, from scratch.

    python scripts/experiments/num_measurements.py --set tuning --steps 2 4 8 16 32
    python scripts/experiments/num_measurements.py --seeds 0 1 2 --steps 10 20 40 80 160

Meant for a simple environment (an empty room, a single obstacle), where the
effect of the number of measurements is easiest to see.
"""
import os
import sys
import dataclasses
import functools
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from methods.registry import AVAILABLE_METHODS
from utils.scenario_sets import add_case_arguments, get_cases
from utils.sim_utils import MeasurementBatch, generate_simulation_data
from utils.plot_utils import set_publication_style
from utils.experiment_utils import run_methods, save_experiment_results, yield_runs


def first_measurements(batch: MeasurementBatch, k: int, seed: int | None) -> MeasurementBatch:
    """Beams of the first k poses, or the first k beams in a fixed random order without poses."""
    if batch.pose_ids is not None:
        mask = batch.pose_ids < k
        return MeasurementBatch(batch.beams[mask], batch.measurements[mask], batch.pose_ids[mask])
    order = torch.as_tensor(np.random.default_rng(seed).permutation(len(batch.measurements))[:k],
                            device=batch.beams.device)
    return MeasurementBatch(batch.beams[order], batch.measurements[order])


def run_step(cfg: Config, k: int, methods: list[str]) -> dict:
    batch, environment = generate_simulation_data(cfg)
    subset = first_measurements(batch, k, cfg.sim.seed)
    metrics, _ = run_methods(subset, environment, cfg, methods)
    return {"metrics": metrics, "beams": int(subset.measurements.shape[0])}


def main():
    parser = ArgumentParser(description="Reconstruction error as measurements are added")
    parser.add_arguments(Config, dest="cfg")
    add_case_arguments(parser)
    parser.add_argument("--steps", nargs="+", type=int, required=True,
                        help="Number of poses (GADEN scenarios) or beams (procedural maps) at each step")
    parser.add_argument("--methods", nargs="+", default=list(AVAILABLE_METHODS), choices=list(AVAILABLE_METHODS))
    parser.add_argument("--workers", type=int, default=1, help="Runs in parallel (times become indicative)")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    set_publication_style()

    cases = get_cases(args.set, args.seeds)
    label = "procedural" if args.seeds else args.set
    unit = "beams" if args.seeds else "poses"
    steps = sorted(args.steps)
    if args.seeds:
        # Procedural maps generate num_beams beams; the steps take prefixes of them
        cfg.sim.num_beams = max(steps)

    tasks = [((c.name, k), c.apply(cfg), {"k": k}) for c in cases for k in steps]
    results = {}
    for key, result, _ in tqdm(yield_runs(functools.partial(run_step, methods=args.methods), tasks,
                                          args.workers, warmup=True), total=len(tasks), desc="Runs"):
        results[key] = result

    names = [c.name for c in cases]
    print(f"\n{'':18s}" + "".join(f"{k:>9d}" for k in steps) + f"   ({unit})")
    curves = {}
    for m in args.methods:
        nrmse = [np.mean([results[(n, k)]["metrics"][m]["nrmse"] for n in names]) for k in steps]
        time_s = [np.mean([results[(n, k)]["metrics"][m]["time"] for n in names]) for k in steps]
        curves[m] = {"nrmse": nrmse, "time": time_s}
        print(f"{m:18s}" + "".join(f"{100 * v:8.2f}%" for v in nrmse))
    beams = [np.mean([results[(n, k)]["beams"] for n in names]) for k in steps]

    save_experiment_results(
        {"experiment_name": f"num_measurements_{label}", "cases": names, "steps": steps, "unit": unit,
         "methods": args.methods, "config": dataclasses.asdict(cfg)},
        {"runs": [{"case": n, "step": k, **results[(n, k)]} for n in names for k in steps],
         "curves": curves, "mean_beams": beams},
    )

    fig, (ax_err, ax_time) = plt.subplots(1, 2, figsize=(14, 5))
    for m in args.methods:
        style = AVAILABLE_METHODS[m]["style"]
        kw = dict(color=style["color"], marker=style["marker"], lw=style.get("linewidth", 1.5),
                  ms=style.get("markersize", 6), label=m)
        ax_err.plot(steps, 100 * np.array(curves[m]["nrmse"]), **kw)
        ax_time.plot(steps, curves[m]["time"], **kw)
    for ax, ylabel in [(ax_err, "NRMSE (% of peak)"), (ax_time, "Reconstruction time (s)")]:
        ax.set_xlabel(f"Number of {unit}")
        ax.set_ylabel(ylabel)
        ax.set_xticks(steps)
        ax.grid(True, ls="--", alpha=0.3)
    ax_err.legend()
    fig.suptitle(f"Reconstruction vs. number of measurements ({label}, mean over {len(names)} cases)")
    fig.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..', '..', 'plots', f'num_measurements_{label}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[+] Plot saved in: {save_path}")


if __name__ == "__main__":
    main()
