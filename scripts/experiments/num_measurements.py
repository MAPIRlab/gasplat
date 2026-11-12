"""
Reconstruction error of every method as the number of measurements grows.

On GADEN scenarios the robot stops at a growing number of vantage poses
(--poses), each with the same sweep (--sim.angular_step); every pose is the one
that adds the most new viewing directions (see select_vantage_poses), so fewer
poses are always a subset of more. Alternatively, the poses stay fixed and the
angular resolution of the sweep changes (--angular_steps, in degrees), as one
would scan more finely with the same robot. Plots show the number of poses or
beams, with the other on top. Procedural maps have no poses, so a growing
number of beams (--beams) is taken from a fixed random order instead. Every
method reconstructs each set of measurements from scratch.

    python scripts/experiments/num_measurements.py --set simple --poses 2 3 4 6 8 12 16 --runs 5
    python scripts/experiments/num_measurements.py --set simple --angular_steps 32 16 8 4 2 1 --runs 5
    python scripts/experiments/num_measurements.py --seeds 0 1 2 --beams 10 20 40 80 160

--runs N runs every GADEN case N times with seeds 0..N-1, which only change the
starting heading of each sweep (the beams turn by up to one angular step; poses
are fixed). Curves show the mean and a band of one standard deviation: the
sensitivity of each method to where exactly the beams fall. The seed is set by
--runs (GADEN) or --seeds (procedural maps), so --seed is rejected. Times are
the reconstruction time of each method, with the system matrix of the grid
methods (total_time).

Meant for a simple environment (the empty room with our own sources), where the
effect of the number of measurements is easiest to see.
"""
import os
import sys
import dataclasses
import functools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from methods.registry import AVAILABLE_METHODS
from utils.paths import PLOTS_DIR
from utils.cli import add_methods_argument, add_workers_argument, reject_options
from utils.scenario_sets import add_case_arguments, cases_label, get_cases
from utils.figure_style import apply_plot_style
from utils.experiment_utils import (batch_config, measurements_at, run_methods, save_experiment_results, total_time,
                                    yield_runs)


def run_step(cfg: Config, step: float, axis: str, methods: list[str]) -> dict:
    """One point of the curve: the number of poses or the angular step (GADEN), or the number of beams (procedural)."""
    batch, environment = measurements_at(cfg, step, axis)
    metrics, _ = run_methods(batch, environment, cfg, methods)
    return {"metrics": metrics, "beams": int(batch.measurements.shape[0])}


def main():
    parser = ArgumentParser(description="Reconstruction error as the number of measurements grows")
    parser.add_arguments(Config, dest="cfg")
    add_case_arguments(parser, default_set="simple")
    parser.add_argument("--poses", nargs="+", type=int, default=None,
                        help="Numbers of vantage poses, for GADEN scenarios (sweep step: --sim.angular_step)")
    parser.add_argument("--angular_steps", nargs="+", type=float, default=None,
                        help="Angular steps of the sweep at each pose (deg) with fixed poses, for GADEN scenarios")
    parser.add_argument("--beams", nargs="+", type=int, default=None, help="Numbers of beams, for procedural maps")
    add_methods_argument(parser)
    parser.add_argument("--runs", type=int, default=1, help="Runs per GADEN case, with seeds 0..N-1 (sweep headings)")
    add_workers_argument(parser)
    args = parser.parse_args()
    reject_options(parser, {"sim.seed": "set by --runs (GADEN scenarios) or --seeds (procedural maps)"})
    cfg: Config = batch_config(args.cfg)
    apply_plot_style()

    cases = get_cases(args.set, args.seeds, cfg.sim.scenario, iteration=args.iteration)
    procedural = cases[0].scenario is None
    axis = "beams" if procedural else ("poses" if args.poses else "angular_step")
    steps = {"beams": args.beams, "poses": args.poses, "angular_step": args.angular_steps}[axis]
    if not steps or (not procedural and args.poses and args.angular_steps):
        sys.exit("Give --beams for procedural maps, or either --poses or --angular_steps for GADEN scenarios")
    steps = sorted(steps, reverse=axis == "angular_step")   # fewest measurements first
    if procedural:
        cfg.sim.num_beams = max(steps)
    label = "procedural" if args.seeds else f"{cases_label(args.set, args.seeds, cfg.sim.scenario, args.iteration)}_{axis}"

    if procedural and args.runs > 1:
        sys.exit("--runs applies to GADEN scenarios; procedural maps vary with --seeds")
    run_seeds = list(range(args.runs))
    tasks = []
    for c in cases:
        for r in run_seeds:
            case_cfg = c.apply(cfg)
            if not procedural:
                case_cfg.sim.seed = r
            tasks.extend(((c.name, s, r), case_cfg, {"step": s}) for s in steps)
    results = {}
    for key, result, _ in tqdm(yield_runs(functools.partial(run_step, axis=axis, methods=args.methods), tasks,
                                          args.workers, warmup=True), total=len(tasks), desc="Runs"):
        results[key] = result

    names = [c.name for c in cases]
    # Procedural seeds share the same numbers of beams and are averaged; GADEN scenarios are shown one by one
    panels = {"mean over seeds": names} if procedural else {n: [n] for n in names}
    runs_of = lambda members, s: [results[(n, s, r)] for n in members for r in run_seeds]
    curves = {}
    for panel, members in panels.items():
        beams = [float(np.mean([x["beams"] for x in runs_of(members, s)])) for s in steps]
        curves[panel] = {"beams": beams}
        print(f"\n{panel}")
        print(f"  {'':18s}" + "".join(f"{b:>9.0f}" for b in beams) + "   (beams)")
        if not procedural:
            unit = "°" if axis == "angular_step" else " "
            print(f"  {'':18s}" + "".join(f"{s:>8g}{unit}" for s in steps) + f"   ({axis.replace('_', ' ')})")
        for m in args.methods:
            values = lambda f: [[f(x["metrics"][m]) for x in runs_of(members, s)] for s in steps]
            nrmse, time_s = values(lambda x: x["nrmse"]), values(total_time)
            curves[panel][m] = {"nrmse": [float(np.mean(v)) for v in nrmse], "nrmse_std": [float(np.std(v)) for v in nrmse],
                                "time": [float(np.mean(v)) for v in time_s], "time_std": [float(np.std(v)) for v in time_s]}
            print(f"  {m:18s}" + "".join(f"{100 * np.mean(v):6.2f}±{100 * np.std(v):4.2f}" for v in nrmse))

    save_experiment_results(
        {"experiment_name": f"num_measurements_{label}", "cases": names, "steps": steps, "axis": axis,
         "step_unit": {"beams": "beams", "poses": "poses", "angular_step": "angular step (deg)"}[axis],
         "case_settings": {c.name: {"scenario": c.scenario and os.path.abspath(c.scenario), "seed": c.seed} for c in cases},
         "methods": args.methods, "iteration": args.iteration,
         "config": dataclasses.asdict(cfg)},
        {"runs": [{"case": n, "step": s, "seed": r, **results[(n, s, r)]} for n in names for s in steps for r in run_seeds],
         "curves": curves, "repeats": len(run_seeds)},
    )

    fig, axs = plt.subplots(2, len(panels), figsize=(7 * len(panels), 9), squeeze=False)
    for j, (panel, curve) in enumerate(curves.items()):
        # x axis: number of poses on the pose axis, number of beams otherwise; the other one on top
        x = [float(s) for s in steps] if axis == "poses" else curve["beams"]
        for m in args.methods:
            style = AVAILABLE_METHODS[m]["style"]
            kw = dict(color=style["color"], marker=style["marker"], lw=style.get("linewidth", 1.5),
                      ms=style.get("markersize", 6), label=m)
            for i, key, factor in [(0, "nrmse", 100), (1, "time", 1)]:
                mean, std = factor * np.array(curve[m][key]), factor * np.array(curve[m][key + "_std"])
                axs[i, j].plot(x, mean, **kw)
                axs[i, j].fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15)
        for i, ylabel in enumerate(["NRMSE (% of peak)", "Reconstruction time (s)"]):
            ax = axs[i, j]
            ax.set_xscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:.0f}" for v in x])
            ax.minorticks_off()
            ax.set_xlabel("Number of poses" if axis == "poses" else "Number of beams")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.3)
        title = panel.split("_h")[0]
        if not procedural:
            top = axs[0, j].secondary_xaxis("top")
            top.set_xticks(x)
            if axis == "poses":
                top.set_xticklabels([f"{b:.0f}" for b in curve["beams"]])
                top.set_xlabel(f"Number of beams (one every {cfg.sim.angular_step:g}° at each pose)")
            else:
                top.set_xticklabels([f"{s:g}°" for s in steps])
                top.set_xlabel("Angular step at each pose")
        axs[0, j].set_title(title)
    axs[0, 0].legend()
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'num_measurements_{label}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[+] Plot saved in: {save_path}")


if __name__ == "__main__":
    main()
