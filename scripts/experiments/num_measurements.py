"""
Reconstruction error of every method as the number of measurements grows.

On GADEN scenarios the vantage poses stay fixed and the angular resolution of
the sweep at each pose changes (--angular_steps, in degrees), as one would scan
more finely with the same robot; the resulting number of beams is what the
plots show, with the angular step on top. Procedural maps have no poses, so a
growing number of beams (--beams) is taken from a fixed random order instead.
Every method reconstructs each set of measurements from scratch.

    python scripts/experiments/num_measurements.py --set simple --angular_steps 32 16 8 4 2 1 --repeats 5
    python scripts/experiments/num_measurements.py --seeds 0 1 2 --beams 10 20 40 80 160

--repeats N runs every GADEN case N times with seeds 0..N-1, which only change the
starting heading of each sweep (the beams turn by up to one angular step; poses
are fixed). Curves show the mean and a band of one standard deviation: the
sensitivity of each method to where exactly the beams fall.

Meant for simple environments (an empty room, a single obstacle), where the
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


def run_step(cfg: Config, step: float, methods: list[str]) -> dict:
    """One point of the curve: the angular step (GADEN) or the number of beams (procedural)."""
    if cfg.sim.scenario:
        cfg.sim.angular_step = step
        batch, environment = generate_simulation_data(cfg)
    else:
        cfg.sim.num_beams = max(cfg.sim.num_beams, int(step))
        batch, environment = generate_simulation_data(cfg)
        order = torch.as_tensor(np.random.default_rng(cfg.sim.seed).permutation(len(batch.measurements))[:int(step)],
                                device=batch.beams.device)
        batch = MeasurementBatch(batch.beams[order], batch.measurements[order])
    metrics, _ = run_methods(batch, environment, cfg, methods)
    return {"metrics": metrics, "beams": int(batch.measurements.shape[0])}


def main():
    parser = ArgumentParser(description="Reconstruction error as the number of measurements grows")
    parser.add_arguments(Config, dest="cfg")
    add_case_arguments(parser, default_set="simple")
    parser.add_argument("--angular_steps", nargs="+", type=float, default=None,
                        help="Angular steps of the sweep at each pose (deg), for GADEN scenarios")
    parser.add_argument("--beams", nargs="+", type=int, default=None, help="Numbers of beams, for procedural maps")
    parser.add_argument("--methods", nargs="+", default=list(AVAILABLE_METHODS), choices=list(AVAILABLE_METHODS))
    parser.add_argument("--repeats", type=int, default=1, help="Runs per GADEN case, with seeds 0..N-1 (sweep headings)")
    parser.add_argument("--workers", type=int, default=1, help="Runs in parallel (times become indicative)")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    set_publication_style()

    cases = get_cases(args.set, args.seeds)
    procedural = cases[0].scenario is None
    steps = args.beams if procedural else args.angular_steps
    if not steps:
        sys.exit("Give --beams for procedural maps or --angular_steps for GADEN scenarios")
    steps = sorted(steps, reverse=not procedural)   # fewest measurements first
    if procedural:
        cfg.sim.num_beams = max(steps)
    label = "procedural" if args.seeds else args.set

    if procedural and args.repeats > 1:
        sys.exit("--repeats applies to GADEN scenarios; procedural maps vary with --seeds")
    repeats = list(range(args.repeats)) if args.repeats > 1 else [cfg.sim.seed]
    tasks = []
    for c in cases:
        for r in repeats:
            case_cfg = c.apply(cfg)
            if not procedural:
                case_cfg.sim.seed = r
            tasks.extend(((c.name, s, r), case_cfg, {"step": s}) for s in steps)
    results = {}
    for key, result, _ in tqdm(yield_runs(functools.partial(run_step, methods=args.methods), tasks,
                                          args.workers, warmup=True), total=len(tasks), desc="Runs"):
        results[key] = result

    names = [c.name for c in cases]
    # Procedural seeds share the same numbers of beams and are averaged; GADEN scenarios are shown one by one
    panels = {"mean over seeds": names} if procedural else {n: [n] for n in names}
    runs_of = lambda members, s: [results[(n, s, r)] for n in members for r in repeats]
    curves = {}
    for panel, members in panels.items():
        beams = [float(np.mean([x["beams"] for x in runs_of(members, s)])) for s in steps]
        curves[panel] = {"beams": beams}
        print(f"\n{panel}")
        print(f"  {'':18s}" + "".join(f"{b:>9.0f}" for b in beams) + "   (beams)")
        if not procedural:
            print(f"  {'':18s}" + "".join(f"{s:>8g}°" for s in steps) + "   (angular step)")
        for m in args.methods:
            values = lambda key: [[x["metrics"][m][key] for x in runs_of(members, s)] for s in steps]
            nrmse, time_s = values("nrmse"), values("time")
            curves[panel][m] = {"nrmse": [float(np.mean(v)) for v in nrmse], "nrmse_std": [float(np.std(v)) for v in nrmse],
                                "time": [float(np.mean(v)) for v in time_s], "time_std": [float(np.std(v)) for v in time_s]}
            print(f"  {m:18s}" + "".join(f"{100 * np.mean(v):6.2f}±{100 * np.std(v):4.2f}" for v in nrmse))

    save_experiment_results(
        {"experiment_name": f"num_measurements_{label}", "cases": names, "steps": steps,
         "step_unit": "beams" if procedural else "angular step (deg)", "methods": args.methods,
         "config": dataclasses.asdict(cfg)},
        {"runs": [{"case": n, "step": s, "seed": r, **results[(n, s, r)]} for n in names for s in steps for r in repeats],
         "curves": curves, "repeats": len(repeats)},
    )

    fig, axs = plt.subplots(2, len(panels), figsize=(7 * len(panels), 9), squeeze=False)
    for j, (panel, curve) in enumerate(curves.items()):
        for m in args.methods:
            style = AVAILABLE_METHODS[m]["style"]
            kw = dict(color=style["color"], marker=style["marker"], lw=style.get("linewidth", 1.5),
                      ms=style.get("markersize", 6), label=m)
            for i, key, factor in [(0, "nrmse", 100), (1, "time", 1)]:
                mean, std = factor * np.array(curve[m][key]), factor * np.array(curve[m][key + "_std"])
                axs[i, j].plot(curve["beams"], mean, **kw)
                axs[i, j].fill_between(curve["beams"], mean - std, mean + std, color=style["color"], alpha=0.15)
        for i, ylabel in enumerate(["NRMSE (% of peak)", "Reconstruction time (s)"]):
            ax = axs[i, j]
            ax.set_xscale("log")
            ax.set_xticks(curve["beams"])
            ax.set_xticklabels([f"{b:.0f}" for b in curve["beams"]])
            ax.minorticks_off()
            ax.set_xlabel("Number of beams")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.3)
        title = panel.split("_h")[0]
        if not procedural:
            top = axs[0, j].secondary_xaxis("top")
            top.set_xticks(curve["beams"])
            top.set_xticklabels([f"{s:g}°" for s in steps])
            top.set_xlabel("Angular step at each pose")
        axs[0, j].set_title(title)
    axs[0, 0].legend()
    fig.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..', '..', 'plots', f'num_measurements_{label}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[+] Plot saved in: {save_path}")


if __name__ == "__main__":
    main()
