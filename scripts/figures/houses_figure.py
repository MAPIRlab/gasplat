"""
Paper figures of the comparison in the houses, from the random-pose runs of
scripts/experiments/random_poses.py:

    python scripts/figures/houses_figure.py [--log results/random_poses_all.jsonl] \\
        [--examples house03 house27] [--run N] [--copy_to ~/paper/figs]

houses_errors: NRMSE over all the free space and over each region of it: covered from at least
    two directions ("2+ directions"), seen along a single direction, and the unobserved space split by the ground
    truth into where there is no gas (false gas: the error is gas placed there) and where there
    is gas (missing gas). Regional NRMSEs are averaged over the runs whose region holds at least
    MIN_AREA of the free cells.
houses_metrics: peak error, peak ratio, peak location error and computation time.
houses_examples: one run of two houses (--examples; the median run of each house, whose GASPLAT
    minus LFD NRMSE is the median of its runs, or --run): ground truth with the poses, the
    regions, and the maps of LFD and GASPLAT with their error in every region.

Bars are the mean over the houses of each house's mean over its runs; dots are the houses.
The bar figures also show GASPLAT without the obstacle penalty (lambda_obs = 0), set apart as an
ablation; the examples do not.
"""
import os
import sys
import json
import shutil
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.abspath(ROOT))
from utils.metrics import gas_map_metrics
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
})
METHODS = ["SART", "RBF Coupled SART", "LFD", "LTD", "Gas Splatting"]
NO_PENALTY = "Gas Splatting (no penalty)"
LABEL = {"RBF Coupled SART": "RBF-SART", "Gas Splatting": "GASPLAT", NO_PENALTY: r"GASPLAT ($\lambda_{\mathrm{obs}}=0$)"}
SHORT = {"SART": "SART", "RBF Coupled SART": "RBF-SART", "LFD": "LFD", "LTD": "LTD", "Gas Splatting": "GASPLAT",
         NO_PENALTY: "GASPLAT\n" + r"($\lambda_{\mathrm{obs}}\!=\!0$)"}
COLOR = {"SART": "tab:orange", "RBF Coupled SART": "tab:green", "LFD": "tab:red", "LTD": "tab:purple",
         "Gas Splatting": "tab:blue", NO_PENALTY: "#6baed6"}
GAS_CMAP = LinearSegmentedColormap.from_list(
    "gas", [(0.0, (1, 1, 1, 1)), (0.05, (1, 0.97, 0.55, 1)), (0.5, (0.98, 0.82, 0.15, 1)), (1.0, (0.90, 0.45, 0.05, 1))])
WALL_GREY = "0.6"
CLEAN_LEVEL = 0.01      # as utils/experiment_utils.py: no true gas below this share of the peak
MIN_AREA = 0.01         # as scripts/experiments/random_poses.py
CROP_LEVEL, CROP_PAD = 0.02, 0.5
# Regions of the free space: covered, single direction, unobserved without gas (false gas), with gas (missing)
REGION_NAMES = ["covered", "single", "false", "missing"]
REGION_LABEL = {"covered": "2+ directions", "single": "Single direction",
                "false": "Unobserved, no gas", "missing": "Unobserved, gas"}
REGION_COLOR = {"covered": "#9ccf8f", "single": "#f2d16b", "false": "#c9c9e8", "missing": "#d6604d"}
GRID = dict(color="0.88", lw=0.6, zorder=0)


def bar_style(m):
    if m == NO_PENALTY:
        return dict(facecolor="white", edgecolor=COLOR[m], hatch="////", lw=1.0)
    return dict(color=COLOR[m], alpha=0.8 if m == "Gas Splatting" else 0.55)


def load(log):
    by_house = defaultdict(list)
    with open(log) as f:
        for line in f:
            r = json.loads(line)
            by_house[r["case"].split("_")[0]].append(r)
    return dict(sorted(by_house.items()))


def house_values(by_house, m, fn, region=None):
    """Each house's mean over its runs (NaN if no run qualifies)."""
    out = []
    for rs in by_house.values():
        v = [fn(r["metrics"][m]) for r in rs
             if region is None or r["metrics"][m].get(f"area_{region}", 0) >= MIN_AREA]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)


def bars(ax, by_house, methods, fn, region=None, log=False, ref=None):
    rng = np.random.default_rng(0)
    values = [house_values(by_house, m, fn, region) for m in methods]
    top = max(np.nanmax(v) for v in values)
    # Room above the data for the values, written at the top of the panel
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.008, top * 6)
    else:
        ax.set_ylim(0, top * 1.3)
    xs = [i + (0.6 if m == NO_PENALTY else 0.0) for i, m in enumerate(methods)]
    if NO_PENALTY in methods:
        ax.axvline(xs[methods.index(NO_PENALTY)] - 0.8, color="0.55", lw=0.8, ls=":", zorder=1)
    for i, m in zip(xs, methods):
        v = house_values(by_house, m, fn, region)
        v = v[~np.isnan(v)]
        mean = v.mean() if len(v) else np.nan
        ax.bar(i, mean, width=0.72, zorder=2, **bar_style(m))
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color="k", lw=0, zorder=3)
        text = f"{mean:.2f}" if mean < 10 else f"{mean:.1f}"
        ax.text(i, 0.985, text, transform=ax.get_xaxis_transform(), ha="center", va="top", rotation=90,
                fontsize=6.5, color="0.15")
    if ref is not None:
        ax.axhline(ref, color="0.3", lw=0.9, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in methods], rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
    ax.grid(axis="y", **GRID)


def errors_figure(by_house, methods, out):
    panels = [("(a) All free space", lambda x: 100 * x["nrmse"], None),
              ("(b) 2+ directions", lambda x: 100 * x["nrmse_covered"], "covered"),
              ("(c) Single direction", lambda x: 100 * x["nrmse_single"], "single"),
              ("(d) False gas", lambda x: 100 * x["nrmse_false"], "false"),
              ("(e) Missing gas", lambda x: 100 * x["nrmse_missing"], "missing")]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 2.35))
    for ax, (title, fn, region) in zip(axes, panels):
        bars(ax, by_house, methods, fn, region)
        ax.set_title(title, pad=4, fontsize=8.5)
    axes[0].set_ylabel("NRMSE (%)")
    fig.tight_layout(w_pad=0.6)
    save(fig, out)


def metrics_figure(by_house, methods, out):
    panels = [("(a) Peak error (%)", lambda x: 100 * abs(x["peak_ratio"] - 1), False, None),
              ("(b) Peak ratio", lambda x: x["peak_ratio"], False, 1.0),
              ("(c) Peak location error (m)", lambda x: x["peak_distance"], False, None),
              ("(d) Time (s)", lambda x: x["time"] + (x.get("matrix_time") or 0.0), True, None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 2.35))
    for ax, (title, fn, log, ref) in zip(axes, panels):
        bars(ax, by_house, methods, fn, log=log, ref=ref)
        ax.set_title(title, pad=4, fontsize=8.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, out)


def region_map(levels, gt, obstacles):
    """Index into REGION_NAMES per free cell, -1 on obstacles."""
    peak = gt[~obstacles].max()
    out = np.full(gt.shape, -1)
    out[levels == 2] = 0
    out[levels == 1] = 1
    out[(levels == 0) & (gt < CLEAN_LEVEL * peak)] = 2
    out[(levels == 0) & (gt >= CLEAN_LEVEL * peak)] = 3
    out[obstacles] = -1
    return out


def crop_box(masks, shape, cell=0.1):
    ys, xs = np.nonzero(np.logical_or.reduce(masks))
    H, W = shape
    pad = round(CROP_PAD / cell)
    return max(0, xs.min() - pad), min(W, xs.max() + 1 + pad), max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)


def median_run(runs, a="Gas Splatting", b="LFD"):
    """The run whose NRMSE difference between methods a and b is the median of the house's runs."""
    diff = {r["run"]: r["metrics"][a]["nrmse"] - r["metrics"][b]["nrmse"] for r in runs}
    order = sorted(diff, key=diff.get)
    return order[len(order) // 2]


def run_maps(maps_dir, case, run):
    """Maps of one run: saved by the experiment, or regenerated from its seed (and saved)."""
    path = os.path.join(maps_dir, f"{case}_run{run}.npz")
    if not os.path.exists(path):
        sys.path.append(os.path.abspath(os.path.join(ROOT, "scripts", "experiments")))
        sys.path.append(os.path.abspath(ROOT))
        from random_poses import run_case
        from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
        from methods.registry import AVAILABLE_METHODS
        from utils.gaden_utils import DEFAULT_ROOT
        cfg = Config(EnvironmentParams(), SimulationParams(), TrainParams(), DensificationParams())
        cfg.quiet, cfg.sim.random_poses, cfg.sim.scenario, cfg.sim.seed = True, True, os.path.join(DEFAULT_ROOT, case), run
        maps = run_case(cfg, list(AVAILABLE_METHODS), keep_maps=True)["maps"]
        os.makedirs(maps_dir, exist_ok=True)
        np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in maps.items()})
    return np.load(path)


def examples_figure(log, by_house, houses, run, out, shown=("LFD", "Gas Splatting")):
    """run None: the median run of each house (median_run)."""
    maps_dir = log.replace(".jsonl", "_maps")
    rows = []
    for h in houses:
        rr = run if run is not None else median_run(by_house[h])
        r = next(x for x in by_house[h] if x["run"] == rr)
        d = run_maps(maps_dir, r["case"], rr)
        print(f"{h}: run {rr}, GASPLAT {100 * r['metrics']['Gas Splatting']['nrmse']:.2f}%, "
              f"LFD {100 * r['metrics']['LFD']['nrmse']:.2f}% in the experiment's log")
        obs = d["obstacles"] > 0.5
        gt = d["gt"]
        peak = gt[~obs].max()
        box = crop_box([(img > CROP_LEVEL * peak) & ~obs for img in (gt, *(d[f"pred/{m}"] for m in shown))], gt.shape)
        rows.append(dict(name=h, d=d, r=r, run=rr, obs=obs, gt=gt, box=box,
                         regions=region_map(d["levels"], gt, obs)))
    ratios = [(x["box"][3] - x["box"][2]) / (x["box"][1] - x["box"][0]) for x in rows]
    ncol = 2 + len(shown)
    map_w = 7.16 / ncol
    fig = plt.figure(figsize=(7.16, sum(map_w * q + 0.62 for q in ratios) + 0.3))
    grid = fig.add_gridspec(len(rows), ncol, height_ratios=[q + 0.62 / map_w for q in ratios], wspace=0.05,
                            hspace=0.08)
    region_cmap = ListedColormap([REGION_COLOR[k] for k in REGION_NAMES])
    letters = iter("abcdefghijkl")
    for i, x in enumerate(rows):
        x0, x1, y0, y1 = x["box"]
        extent = (x0 * 0.1, x1 * 0.1, y0 * 0.1, y1 * 0.1)
        crop = lambda a: a[y0:y1, x0:x1]
        obs = crop(x["obs"])
        vmax = x["gt"][~x["obs"]].max()
        P = x["d"]["poses"]
        inside = (P[:, 0] >= extent[0]) & (P[:, 0] <= extent[1]) & (P[:, 1] >= extent[2]) & (P[:, 1] <= extent[3])
        for j in range(ncol):
            ax = fig.add_subplot(grid[i, j])
            if j == 1:
                reg = crop(x["regions"]).astype(float)
                ax.imshow(np.where(obs, np.nan, reg), origin="lower", extent=extent, cmap=region_cmap,
                          vmin=-0.5, vmax=3.5, interpolation="nearest")
                title = "Regions"
            else:
                img = x["gt"] if j == 0 else x["d"][f"pred/{shown[j - 2]}"]
                ax.imshow(np.where(obs, np.nan, crop(img)), origin="lower", extent=extent, cmap=GAS_CMAP, vmin=0,
                          vmax=vmax, interpolation="bilinear")
                title = f"{x['name'].capitalize()}: ground truth" if j == 0 else LABEL.get(shown[j - 2], shown[j - 2])
            ax.imshow(np.where(obs, 1.0, np.nan), origin="lower", extent=extent, cmap=ListedColormap([WALL_GREY]),
                      interpolation="nearest")
            if j < 2:
                ax.scatter(P[inside, 0], P[inside, 1], marker="^", s=20, c="w", edgecolors="k", linewidths=0.6,
                           zorder=3)
            if j >= 2:
                m = gas_map_metrics(x["gt"], x["d"][f"pred/{shown[j - 2]}"], x["d"]["obstacles"], 0.1, x["d"]["levels"])
                fmt = lambda k: "–" if np.isnan(m.get(k, np.nan)) else f"{100 * m[k]:.1f}"   # – where the region is empty
                ax.text(0.5, -0.03, f"all {100 * m['nrmse']:.1f}, 2+ dir. {fmt('nrmse_covered')}, single {fmt('nrmse_single')}\n"
                        f"false {fmt('nrmse_false')}, missing {fmt('nrmse_missing')} (NRMSE %)",
                        transform=ax.transAxes, ha="center", va="top", fontsize=7)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(f"({next(letters)}) {title}", pad=3, fontsize=8.5)
    fig.legend(handles=[Patch(color=REGION_COLOR[k], label=REGION_LABEL[k]) for k in REGION_NAMES],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    save(fig, out)


def save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description="Paper figures of the comparison in the houses")
    ap.add_argument("--log", default=os.path.join(ROOT, "results", "random_poses_all.jsonl"))
    ap.add_argument("--examples", nargs=2, default=["house03", "house27"])
    ap.add_argument("--run", type=int, default=None,
                    help="Run shown in the examples (default: the median run of each house, GASPLAT minus LFD)")
    ap.add_argument("--output_dir", default=os.path.join(ROOT, "plots"))
    ap.add_argument("--copy_to", default=None, help="Also copy the PDFs here (e.g. ~/paper/figs)")
    args = ap.parse_args()
    by_house = load(args.log)
    methods = METHODS + [NO_PENALTY]
    outs = [os.path.join(args.output_dir, n) for n in ("houses_errors", "houses_metrics", "houses_examples")]
    errors_figure(by_house, methods, outs[0])
    metrics_figure(by_house, methods, outs[1])
    examples_figure(args.log, by_house, args.examples, args.run, outs[2])
    print(f"{len(by_house)} houses, runs per house: {sorted({len(v) for v in by_house.values()})}")
    if args.copy_to:
        for o in outs:
            shutil.copy(f"{o}.pdf", os.path.expanduser(args.copy_to))


if __name__ == "__main__":
    main()
