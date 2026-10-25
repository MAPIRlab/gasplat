"""Paper figure of the experiment on the number of measurements, from its results.

Top row: one case of the experiment at one of its steps (--step): (a) the ground
truth with the beams (faint, so that the ground truth stays visible) and the
poses, (b) GASPLAT, (c) the strongest baseline (--baseline, default LFD), all from
the same measurements, regenerated with the configuration of the experiment.
Bottom row: (d) NRMSE and (e) computation time of every method along the axis of
the experiment (poses, angular step or beams), mean and one standard deviation
over the sweep headings (over the seeds for procedural maps). The time of a grid
method includes the construction of its system matrix, so every time covers the
whole reconstruction from the measurements to the map.

Paper figure (side plume, 4 poses):
    python scripts/experiments/num_measurements.py --scenario data/scenarios/10x6_empty_room_side_plume_h0.75_t1000 \\
        --poses 2 3 4 6 8 12 16 --repeats 5 --workers 1
    python scripts/figures/num_measurements_figure.py \\
        results/num_measurements_10x6_empty_room_side_plume_h0.75_t1000_poses_<date>.json --step 4 \\
        --output plots/num_measurements_side_plume
"""
import os
import sys
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
from methods.registry import AVAILABLE_METHODS
from utils.gaden_utils import DEFAULT_ROOT
from utils.scenario_sets import Case
from utils.experiment_utils import measurements_at, run_methods

# STIX (Times-like): heavier strokes than Computer Modern, readable at the printed size
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

GAS_CMAP = LinearSegmentedColormap.from_list(
    "gas", [(0.0, (1, 1, 1, 1)), (0.05, (1, 0.97, 0.55, 1)), (0.5, (0.98, 0.82, 0.15, 1)), (1.0, (0.90, 0.45, 0.05, 1))])
WALL_GREY = "0.6"
BEAM_COLOR = "0.55"    # beams of the ground-truth panel: light and faint, so that the plume stays visible
BEAM_ALPHA = 0.15
NAMES = {"Gas Splatting": r"\method", "RBF Coupled SART": "RBF-SART"}   # labels used in the paper
ORDER = ["SART", "RBF Coupled SART", "LFD", "LTD", "Gas Splatting"]
# Axis of the experiment, for results saved before the metadata had "axis"
AXES = {"poses": "poses", "beams": "beams", "angular step (deg)": "angular_step"}


def label(method: str) -> str:
    return NAMES.get(method, method).replace(r"\method", "GASPLAT")


def step_text(step: float, axis: str) -> str:
    return {"poses": f"{step:g} poses", "angular_step": f"{step:g}° step", "beams": f"{step:g} beams"}[axis]


def experiment_config(metadata: dict) -> Config:
    """The configuration the experiment ran with."""
    c = metadata["config"]
    env = EnvironmentParams(**{**c["env"], "map_size": tuple(c["env"]["map_size"])})
    return Config(env, SimulationParams(**c["sim"]), TrainParams(**c["train"]), DensificationParams(**c["densify"]),
                  quiet=True, device_type=c["device_type"])


def case_of(metadata: dict, name: str) -> Case:
    """A case of the experiment. Results saved before "case_settings" only had GADEN scenarios in data/scenarios."""
    settings = metadata.get("case_settings", {}).get(name, {"scenario": os.path.join(DEFAULT_ROOT, name), "seed": None})
    return Case(name, **settings)


def curves(data: dict, members: list[str]):
    """Mean and std of NRMSE and total time (with the matrix) per method and step, over the runs of the members."""
    steps, runs = data["metadata"]["steps"], [r for r in data["results"]["runs"] if r["case"] in members]
    out = {}
    for m in data["metadata"]["methods"]:
        nrmse, time = [], []
        for s in steps:
            rs = [r["metrics"][m] for r in runs if r["step"] == s]
            nrmse.append([r["nrmse"] for r in rs])
            time.append([r["time"] + (r.get("matrix_time") or 0.0) for r in rs])
        out[m] = {k: (np.mean(v, axis=1), np.std(v, axis=1)) for k, v in (("nrmse", nrmse), ("time", time))}
    beams = [np.mean([r["beams"] for r in runs if r["step"] == s]) for s in steps]
    return np.array(steps, dtype=float), np.array(beams), out


def draw_map(ax, img, obstacles, extent, vmax, title):
    ax.imshow(np.where(obstacles, np.nan, img), origin="lower", extent=extent, cmap=GAS_CMAP, vmin=0, vmax=vmax,
              interpolation="bilinear")
    ax.imshow(np.where(obstacles, 1.0, np.nan), origin="lower", extent=extent, cmap=ListedColormap([WALL_GREY]),
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, pad=3)


def main():
    parser = argparse.ArgumentParser(description="Figure of the experiment on the number of measurements")
    parser.add_argument("results", help="Results of scripts/experiments/num_measurements.py (JSON)")
    parser.add_argument("--case", default=None, help="Case shown in the top row (default: the first of the experiment)")
    parser.add_argument("--step", type=float, default=None,
                        help="Step of the experiment shown in the top row (default: the middle one)")
    parser.add_argument("--baseline", default="LFD", choices=[m for m in AVAILABLE_METHODS if m != "Gas Splatting"])
    parser.add_argument("--seed", type=int, default=0, help="Sweep headings of the reconstructions shown (GADEN)")
    parser.add_argument("--output", default=None, help="Output path without extension (default: plots/<experiment name>)")
    parser.add_argument("--copy_to", default=None, help="Also copy the PDF here (e.g. the paper's figs folder)")
    args = parser.parse_args()

    data = json.load(open(args.results))
    meta = data["metadata"]
    axis = meta.get("axis") or AXES[meta["step_unit"]]
    steps = meta["steps"]
    step = args.step if args.step is not None else steps[len(steps) // 2]
    if step not in steps:
        sys.exit(f"--step must be one of the steps of the experiment: {steps}")
    case = case_of(meta, args.case or meta["cases"][0])
    if case.name not in meta["cases"]:
        sys.exit(f"--case must be one of the cases of the experiment: {meta['cases']}")
    procedural = case.scenario is None

    # --- Reconstructions at one step ---
    cfg = case.apply(experiment_config(meta))
    if not procedural:
        cfg.sim.seed = args.seed
    batch, environment = measurements_at(cfg, step, axis)
    metrics, preds = run_methods(batch, environment, cfg, ["Gas Splatting", args.baseline])
    gt = environment.ground_truth.gas_map
    obstacles = environment.obstacles > 0.5 if environment.obstacles is not None else np.zeros(gt.shape, dtype=bool)
    H, W = gt.shape
    extent = (0, W * cfg.env.cell_size, 0, H * cfg.env.cell_size)
    vmax = max(gt[~obstacles].max(), *(p[~obstacles].max() for p in preds.values()))
    beams = batch.beams.cpu().numpy()

    # --- Curves: the case alone, or the mean over the seeds of procedural maps ---
    x_steps, n_beams, curve = curves(data, meta["cases"] if procedural else [case.name])
    # x axis: number of poses on the pose axis, number of beams otherwise
    x = x_steps if axis == "poses" else n_beams
    marker = x[steps.index(step)]

    fig = plt.figure(figsize=(7.16, 4.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.38, left=0.02, right=0.98, top=0.95, bottom=0.10)
    top_row = grid[0].subgridspec(1, 3, wspace=0.12)
    bottom_row = grid[1].subgridspec(1, 3, wspace=0.42, width_ratios=[1, 1, 0.55])
    maps = [fig.add_subplot(top_row[0, j]) for j in range(3)]
    draw_map(maps[0], gt, obstacles, extent, vmax, f"(a) Ground truth, {step_text(step, axis)}")
    maps[0].add_collection(LineCollection(beams, colors=BEAM_COLOR, linewidths=0.3, alpha=BEAM_ALPHA))
    if batch.pose_ids is not None:
        ids = batch.pose_ids.cpu().numpy()
        poses = np.array([beams[ids == p][0, 0] for p in np.unique(ids)])   # common origin of each pose's beams
        maps[0].scatter(poses[:, 0], poses[:, 1], marker="^", s=28, c="w", edgecolors="k", linewidths=0.6, zorder=3)
    for ax, (m, letter) in zip(maps[1:], (("Gas Splatting", "b"), (args.baseline, "c"))):
        draw_map(ax, preds[m], obstacles, extent, vmax, f"({letter}) {label(m)}, NRMSE {100 * metrics[m]['nrmse']:.1f}%")
    sm = plt.cm.ScalarMappable(cmap=GAS_CMAP, norm=plt.Normalize(0, vmax))
    cbar = fig.colorbar(sm, ax=maps, fraction=0.015, pad=0.01)
    cbar.set_label("Concentration (ppm)")

    plots = [fig.add_subplot(bottom_row[0, 0]), fig.add_subplot(bottom_row[0, 1])]
    for ax, key, factor, ylabel in ((plots[0], "nrmse", 100, "NRMSE (% of peak)"),
                                    (plots[1], "time", 1, "Computation time (s)")):
        for m in ORDER:
            if m not in curve:
                continue
            style = AVAILABLE_METHODS[m]["style"]
            center, spread = factor * curve[m][key][0], factor * curve[m][key][1]
            lo, hi = np.maximum(center - spread, 0), center + spread
            ax.plot(x, center, color=style["color"], marker=style["marker"], lw=1.8 if m == "Gas Splatting" else 1.1,
                    ms=7 if m == "Gas Splatting" else 4.5, label=label(m))
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.15, lw=0)
        ax.axvline(marker, color="0.5", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:.0f}" for v in x]); ax.minorticks_off()
        ax.set_xlabel("Number of poses" if axis == "poses" else "Number of beams")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.3)
        if axis != "beams":   # the other quantity on top
            top = ax.secondary_xaxis("top")
            top.set_xticks(x); top.minorticks_off()
            if axis == "poses":
                top.set_xticklabels([f"{b:.0f}" for b in n_beams])
                top.set_xlabel("Number of beams", labelpad=2)
            else:
                top.set_xticklabels([f"{s:g}°" for s in steps])
                top.set_xlabel("Angular step at each pose", labelpad=2)
    plots[0].set_ylim(bottom=0)
    plots[1].set_ylim(bottom=0)
    for ax, letter in zip(plots, "de"):
        ax.text(-0.16, 1.17, f"({letter})", transform=ax.transAxes, ha="left", va="bottom")

    # Legend in the free third column
    lax = fig.add_subplot(bottom_row[0, 2]); lax.axis("off")
    lax.legend(*plots[0].get_legend_handles_labels(), loc="center", frameon=False)

    output = args.output or os.path.join("plots", meta["experiment_name"])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{output}.{ext}", dpi=300, bbox_inches="tight")
    if args.copy_to:
        os.makedirs(os.path.expanduser(args.copy_to), exist_ok=True)
        shutil.copy(f"{output}.pdf", os.path.expanduser(args.copy_to))
    print("results:", args.results)
    for m in metrics:
        print(f"{label(m)} on {case.name}, {step_text(step, axis)} (seed {cfg.sim.seed}): "
              f"NRMSE {100 * metrics[m]['nrmse']:.2f}%")
    print(f"Saved {output}.{{pdf,png}}")


if __name__ == "__main__":
    main()
