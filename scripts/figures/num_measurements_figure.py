"""Paper figure of the experiment on the number of vantage poses (side plume, empty room).

Top row: the scenario and two reconstructions from the same measurements at one
number of poses (--poses, default 4): (a) the ground truth with the poses and
their beams, (b) GASPLAT, (c) the strongest baseline (--baseline, default LFD).
Bottom row: (d) NRMSE and (e) computation time of every method over the number of
poses, read from the results of scripts/experiments/num_measurements.py (mean and one
standard deviation over the sweep headings). The time of a grid method
includes the construction of its system matrix, so every time covers the whole
reconstruction from the measurements to the map.

    python scripts/experiments/num_measurements.py --set side_plume --poses 2 3 4 6 8 12 16 --repeats 5 --workers 1
    python scripts/figures/num_measurements_figure.py [--results results/num_measurements_side_plume_poses_*.json]
"""
import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
from methods.registry import AVAILABLE_METHODS
from utils.gaden_utils import DEFAULT_ROOT
from utils.sim_utils import generate_simulation_data
from utils.experiment_utils import run_methods

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

SCENARIO = "10x6_empty_room_side_plume_h0.75_t1000"
GAS_CMAP = LinearSegmentedColormap.from_list(
    "gas", [(0.0, (1, 1, 1, 1)), (0.05, (1, 0.97, 0.55, 1)), (0.5, (0.98, 0.82, 0.15, 1)), (1.0, (0.90, 0.45, 0.05, 1))])
WALL_GREY = "0.6"
NAMES = {"Gas Splatting": r"\method", "RBF Coupled SART": "RBF-SART"}   # labels used in the paper
ORDER = ["SART", "RBF Coupled SART", "LFD", "LTD", "Gas Splatting"]


def label(method: str) -> str:
    return NAMES.get(method, method).replace(r"\method", "GASPLAT")


def latest_results() -> str:
    files = sorted(glob.glob("results/num_measurements_side_plume_poses_*.json"))
    if not files:
        sys.exit("No results: run scripts/experiments/num_measurements.py --set side_plume --poses ... first")
    return files[-1]


def curves(path: str):
    """Mean and std of NRMSE and total time (with the matrix) per method and number of poses."""
    data = json.load(open(path))
    steps, runs = data["metadata"]["steps"], data["results"]["runs"]
    beams = data["results"]["curves"][next(iter(data["results"]["curves"]))]["beams"]
    out = {}
    for m in data["metadata"]["methods"]:
        nrmse, time = [], []
        for s in steps:
            rs = [r["metrics"][m] for r in runs if r["step"] == s]
            nrmse.append([r["nrmse"] for r in rs])
            time.append([r["time"] + (r.get("matrix_time") or 0.0) for r in rs])
        out[m] = {k: (np.mean(v, axis=1), np.std(v, axis=1)) for k, v in (("nrmse", nrmse), ("time", time))}
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
    parser = argparse.ArgumentParser(description="Figure of the experiment on the number of poses")
    parser.add_argument("--results", default=None, help="Results of num_measurements.py (default: the latest)")
    parser.add_argument("--poses", type=int, default=4, help="Number of poses of the reconstructions shown")
    parser.add_argument("--baseline", default="LFD", choices=[m for m in AVAILABLE_METHODS if m != "Gas Splatting"])
    parser.add_argument("--seed", type=int, default=0, help="Sweep headings of the reconstructions shown")
    parser.add_argument("--copy_to", default=None, help="Also copy the PDF here (e.g. the paper's figs folder)")
    args = parser.parse_args()

    # --- Reconstructions at one number of poses ---
    cfg = Config(EnvironmentParams(), SimulationParams(), TrainParams(), DensificationParams())
    cfg.sim.scenario = os.path.join(DEFAULT_ROOT, SCENARIO)
    cfg.sim.num_poses = args.poses
    cfg.sim.seed = args.seed
    cfg.quiet = True
    batch, environment = generate_simulation_data(cfg)
    metrics, preds = run_methods(batch, environment, cfg, ["Gas Splatting", args.baseline])
    gt, obstacles = environment.ground_truth.gas_map, environment.obstacles > 0.5
    H, W = gt.shape
    extent = (0, W * cfg.env.cell_size, 0, H * cfg.env.cell_size)
    vmax = max(gt[~obstacles].max(), *(p[~obstacles].max() for p in preds.values()))
    beams = batch.beams.cpu().numpy()
    ids = batch.pose_ids.cpu().numpy()
    poses = np.array([beams[ids == p][0, 0] for p in np.unique(ids)])   # common origin of each pose's beams

    # --- Curves ---
    path = args.results or latest_results()
    steps, n_beams, curve = curves(path)

    fig = plt.figure(figsize=(7.16, 4.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.38, left=0.02, right=0.98, top=0.95, bottom=0.10)
    top_row = grid[0].subgridspec(1, 3, wspace=0.12)
    bottom_row = grid[1].subgridspec(1, 3, wspace=0.42, width_ratios=[1, 1, 0.55])
    maps = [fig.add_subplot(top_row[0, j]) for j in range(3)]
    draw_map(maps[0], gt, obstacles, extent, vmax, f"(a) Ground truth, {args.poses} poses")
    alpha = float(np.clip(8.0 / np.sqrt(max(len(beams), 1)), 0.15, 0.6))
    maps[0].add_collection(LineCollection(beams, colors="0.35", linewidths=0.4, alpha=alpha))
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
            ax.plot(steps, center, color=style["color"], marker=style["marker"], lw=1.8 if m == "Gas Splatting" else 1.1,
                    ms=7 if m == "Gas Splatting" else 4.5, label=label(m))
            ax.fill_between(steps, lo, hi, color=style["color"], alpha=0.15, lw=0)
        ax.axvline(args.poses, color="0.5", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_xticks(steps); ax.set_xticklabels([f"{s:.0f}" for s in steps]); ax.minorticks_off()
        ax.set_xlabel("Number of poses")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.3)
        top = ax.secondary_xaxis("top")
        top.set_xticks(steps); top.set_xticklabels([f"{b:.0f}" for b in n_beams])
        top.set_xlabel("Number of beams", labelpad=2)
    plots[0].set_ylim(bottom=0)
    plots[0].yaxis.set_major_locator(plt.MultipleLocator(2))
    plots[1].set_ylim(bottom=0)
    for ax, letter in zip(plots, "de"):
        ax.text(-0.16, 1.17, f"({letter})", transform=ax.transAxes, ha="left", va="bottom")

    # Legend in the free third column
    lax = fig.add_subplot(bottom_row[0, 2]); lax.axis("off")
    lax.legend(*plots[0].get_legend_handles_labels(), loc="center", frameon=False)

    os.makedirs("plots", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/num_measurements_side_plume.{ext}", dpi=300, bbox_inches="tight")
    if args.copy_to:
        os.makedirs(os.path.expanduser(args.copy_to), exist_ok=True)
        shutil.copy("plots/num_measurements_side_plume.pdf", os.path.expanduser(args.copy_to))
    print("results:", path)
    for m in metrics:
        print(f"{label(m)} at {args.poses} poses (seed {args.seed}): NRMSE {100 * metrics[m]['nrmse']:.2f}%")
    print("Saved plots/num_measurements_side_plume.{pdf,png}")


if __name__ == "__main__":
    main()
