"""Paper figures of the comparison on the held-out houses.

  metrics:  one bar panel per metric (NRMSE, peak error, peak location error, mass
            error, time): mean over the houses, one standard deviation across
            houses, and every house as a dot. Read from the results of
            scripts/experiments/compare_methods.py (a house counts with the mean
            over its sweep headings; signed errors in absolute value; times of the
            grid methods include the system matrix).
  examples: two houses, one per row: the ground truth with the poses and beams,
            GASPLAT and LFD, from the same measurements (seed 0). Maps are cropped,
            with the same rule in every panel, to the region where the true or a
            reconstructed concentration exceeds a small share of the true peak.

    python scripts/experiments/compare_methods.py --set held_out --repeats 5 --workers 1
    python scripts/figures/compare_methods_figure.py [--results ...] [--houses house16 house27] [--copy_to ~/paper/figs]
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
from methods.registry import AVAILABLE_METHODS
from utils.scenario_sets import scenario_dirs
from utils.sim_utils import generate_simulation_data
from utils.experiment_utils import run_methods

# STIX, as in the other figures of the paper
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

ORDER = ["SART", "RBF Coupled SART", "LFD", "LTD", "Gas Splatting"]
LABELS = {"RBF Coupled SART": "RBF-SART", "Gas Splatting": "GASPLAT"}
METRICS = [("nrmse", 100, "NRMSE (%)"), ("peak_error", 100, "Peak error (%)"),
           ("peak_distance", 1, "Peak location error (m)"), ("mass_error", 100, "Mass error (%)"),
           ("time", 1, "Time (s)")]
GAS_CMAP = LinearSegmentedColormap.from_list(
    "gas", [(0.0, (1, 1, 1, 1)), (0.05, (1, 0.97, 0.55, 1)), (0.5, (0.98, 0.82, 0.15, 1)), (1.0, (0.90, 0.45, 0.05, 1))])
WALL_GREY = "0.6"
CROP_LEVEL = 0.02   # share of the true peak above which a cell counts as holding gas, for the crop
CROP_PAD = 0.5      # m around that region


def label(m):
    return LABELS.get(m, m)


def latest_results():
    files = sorted(glob.glob("results/compare_methods_held_out_*.json"))
    if not files:
        sys.exit("No results: run scripts/experiments/compare_methods.py --set held_out --repeats 5 first")
    return files[-1]


def metrics_figure(path, out):
    data = json.load(open(path))
    per_case = data["results"]["per_case"]
    houses = list(per_case)
    methods = [m for m in ORDER if m in data["metadata"]["methods"]]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.16, 2.3), constrained_layout=True)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.18, 0.18, len(houses))
    x = np.arange(len(methods))
    for ax, (key, scale, ylabel) in zip(axes, METRICS):
        values = np.array([[scale * per_case[h][m][key] for h in houses] for m in methods])   # (methods, houses)
        colors = [AVAILABLE_METHODS[m]["style"]["color"] for m in methods]
        ax.bar(x, values.mean(axis=1), yerr=values.std(axis=1), color=colors, alpha=0.55, width=0.7,
               error_kw=dict(lw=0.8, capsize=2, ecolor="0.25"))
        for i in range(len(methods)):
            ax.scatter(x[i] + jitter, values[i], s=7, color="k", zorder=3, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels([label(m) for m in methods], rotation=50, ha="right", rotation_mode="anchor")
        ax.set_title(ylabel, pad=3)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", ls="--", alpha=0.3)
        ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def crop_box(masks, cell, shape):
    ys, xs = np.nonzero(np.logical_or.reduce(masks))
    H, W = shape
    pad = round(CROP_PAD / cell)
    return (max(0, xs.min() - pad), min(W, xs.max() + 1 + pad), max(0, ys.min() - pad), min(H, ys.max() + 1 + pad))


def examples_figure(houses, baseline, seed, out):
    rows = []
    for h in houses:
        scenario = next(d for d in scenario_dirs("held_out") if os.path.basename(d).startswith(h))
        cfg = Config(EnvironmentParams(), SimulationParams(), TrainParams(), DensificationParams())
        cfg.sim.scenario, cfg.sim.seed, cfg.quiet = scenario, seed, True
        batch, environment = generate_simulation_data(cfg)
        metrics, preds = run_methods(batch, environment, cfg, ["Gas Splatting", baseline])
        gt, obstacles = environment.ground_truth.gas_map, environment.obstacles > 0.5
        peak = gt[~obstacles].max()
        box = crop_box([(img > CROP_LEVEL * peak) & ~obstacles for img in (gt, *preds.values())], cfg.env.cell_size, gt.shape)
        beams = batch.beams.cpu().numpy()
        ids = batch.pose_ids.cpu().numpy()
        poses = np.array([beams[ids == p][0, 0] for p in np.unique(ids)])
        rows.append(dict(name=h, gt=gt, obstacles=obstacles, preds=preds, metrics=metrics, box=box,
                         cell=cfg.env.cell_size, beams=beams, poses=poses))
        print(f"{h}: GASPLAT {100 * metrics['Gas Splatting']['nrmse']:.2f}%, {baseline} {100 * metrics[baseline]['nrmse']:.2f}%")

    # Row heights follow the aspect ratio of each crop
    ratios = [(r["box"][3] - r["box"][2]) / (r["box"][1] - r["box"][0]) for r in rows]
    map_w = 2.15
    fig = plt.figure(figsize=(7.16, sum(map_w * q for q in ratios) + 0.35 * len(rows)))
    grid = fig.add_gridspec(len(rows), 4, height_ratios=ratios, width_ratios=[1, 1, 1, 0.05], wspace=0.06, hspace=0.18)
    for i, r in enumerate(rows):
        x0, x1, y0, y1 = r["box"]
        c = r["cell"]
        extent = (x0 * c, x1 * c, y0 * c, y1 * c)
        crop = lambda a: a[y0:y1, x0:x1]
        obs = crop(r["obstacles"])
        vmax = max(r["gt"][~r["obstacles"]].max(), *(p[~r["obstacles"]].max() for p in r["preds"].values()))
        panels = [(r["gt"], f"{r['name'].capitalize()}: ground truth"),
                  (r["preds"]["Gas Splatting"], f"GASPLAT, NRMSE {100 * r['metrics']['Gas Splatting']['nrmse']:.1f}%"),
                  (r["preds"][baseline], f"{label(baseline)}, NRMSE {100 * r['metrics'][baseline]['nrmse']:.1f}%")]
        for j, (img, title) in enumerate(panels):
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(np.where(obs, np.nan, crop(img)), origin="lower", extent=extent, cmap=GAS_CMAP, vmin=0, vmax=vmax,
                      interpolation="bilinear")
            ax.imshow(np.where(obs, 1.0, np.nan), origin="lower", extent=extent, cmap=ListedColormap([WALL_GREY]),
                      interpolation="nearest")
            if j == 0:
                alpha = float(np.clip(8.0 / np.sqrt(len(r["beams"])), 0.1, 0.6))
                ax.add_collection(LineCollection(r["beams"], colors="0.35", linewidths=0.35, alpha=alpha))
                ax.scatter(r["poses"][:, 0], r["poses"][:, 1], marker="^", s=24, c="w", edgecolors="k", linewidths=0.6, zorder=3)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(f"({'abcdef'[3 * i + j]}) {title}", pad=3)
        cax = fig.add_subplot(grid[i, 3])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=GAS_CMAP, norm=plt.Normalize(0, vmax)), cax=cax)
        cbar.set_label("ppm")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Figures of the comparison on the held-out houses")
    parser.add_argument("--results", default=None, help="Results of compare_methods.py (default: the latest held-out run)")
    parser.add_argument("--houses", nargs="+", default=["house16", "house27"], help="Houses of the examples figure")
    parser.add_argument("--baseline", default="LFD", choices=[m for m in AVAILABLE_METHODS if m != "Gas Splatting"])
    parser.add_argument("--seed", type=int, default=0, help="Sweep headings of the example reconstructions")
    parser.add_argument("--copy_to", default=None, help="Also copy the PDFs here (e.g. the paper's figs folder)")
    args = parser.parse_args()

    os.makedirs("plots", exist_ok=True)
    path = args.results or latest_results()
    metrics_figure(path, "plots/compare_methods_metrics")
    examples_figure(args.houses, args.baseline, args.seed, "plots/compare_methods_examples")
    if args.copy_to:
        os.makedirs(os.path.expanduser(args.copy_to), exist_ok=True)
        for name in ("compare_methods_metrics", "compare_methods_examples"):
            shutil.copy(f"plots/{name}.pdf", os.path.expanduser(args.copy_to))
    print("results:", path)
    print("Saved plots/compare_methods_metrics.{pdf,png} and plots/compare_methods_examples.{pdf,png}")


if __name__ == "__main__":
    main()
