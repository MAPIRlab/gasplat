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
import glob
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT)
from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams, config_from_dict
from methods.registry import AVAILABLE_METHODS
from utils.paths import RESULTS_DIR, PLOTS_DIR
from utils.gaden_utils import DEFAULT_ROOT
from utils.metrics import CLEAN_LEVEL, gas_map_metrics   # no true gas below CLEAN_LEVEL of the peak
from utils.experiment_utils import total_time
from utils.figure_style import (METHOD_ORDER, apply_paper_style, draw_gas_map, method_color, method_label,
                                save_figure)

apply_paper_style(legend_fontsize=8.5, axes_spines_top=False, axes_spines_right=False)

METHODS = METHOD_ORDER
NO_PENALTY = "Gas Splatting (no penalty)"
LABEL = {m: method_label(m) for m in METHODS} | {NO_PENALTY: r"GASPLAT ($\lambda_{\mathrm{obs}}=0$)"}
SHORT = LABEL | {NO_PENALTY: "GASPLAT\n" + r"($\lambda_{\mathrm{obs}}\!=\!0$)"}   # bar labels
COLOR = {m: method_color(m) for m in METHODS} | {NO_PENALTY: "#6baed6"}
MIN_AREA = 0.01         # as scripts/experiments/random_poses.py
CROP_LEVEL, CROP_PAD = 0.02, 0.5
# Regions of the free space: covered, single direction, unobserved without gas (false gas), with gas (missing)
REGION_NAMES = ["covered", "single", "false", "missing"]
REGION_LABEL = {"covered": "2+ directions", "single": "Single direction", "false": "False gas", "missing": "Missing gas"}
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


def errors_figure(by_house, methods, out, copy_to=None):
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
    save_figure(fig, out, copy_to=copy_to)


def metrics_figure(by_house, methods, out, copy_to=None):
    panels = [("(a) Peak error (%)", lambda x: 100 * abs(x["peak_ratio"] - 1), False, None),
              ("(b) Peak ratio", lambda x: x["peak_ratio"], False, 1.0),
              ("(c) Peak location error (m)", lambda x: x["peak_distance"], False, None),
              ("(d) Time (s)", total_time, True, None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 2.35))
    for ax, (title, fn, log, ref) in zip(axes, panels):
        bars(ax, by_house, methods, fn, log=log, ref=ref)
        ax.set_title(title, pad=4, fontsize=8.5)
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, out, copy_to=copy_to)


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


def experiment_config(log):
    """Configuration of the experiment that wrote the log: that of its latest summary
    (scripts/experiments/random_poses.py saves it in metadata.config, with the log in
    metadata.log; summaries are saved next to the log), else the default with a warning."""
    folder = os.path.dirname(os.path.abspath(log))
    for path in sorted(glob.glob(os.path.join(folder, "random_poses_*.json")), reverse=True):   # latest first
        meta = json.load(open(path)).get("metadata", {})
        if os.path.basename(meta.get("log", "")) == os.path.basename(log) and "config" in meta:
            return config_from_dict(meta["config"])
    print(f"Warning: no summary of {log} in {folder}, the maps are regenerated with the default configuration")
    return Config(EnvironmentParams(), SimulationParams(), TrainParams(), DensificationParams())


def run_maps(log, case, run):
    """Maps of one run: saved by the experiment, or regenerated from its seed with the
    configuration of the experiment (and saved)."""
    maps_dir = log.replace(".jsonl", "_maps")
    path = os.path.join(maps_dir, f"{case}_run{run}.npz")
    if not os.path.exists(path):
        sys.path.append(os.path.join(ROOT, "scripts", "experiments"))
        from random_poses import run_case
        cfg = experiment_config(log)
        cfg.quiet, cfg.sim.random_poses, cfg.sim.scenario, cfg.sim.seed = True, True, os.path.join(DEFAULT_ROOT, case), run
        maps = run_case(cfg, list(AVAILABLE_METHODS), keep_maps=True)["maps"]
        os.makedirs(maps_dir, exist_ok=True)
        np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in maps.items()})
    return np.load(path)


def region_table(ax, rows):
    """NRMSE (%) of each method (rows: [(label, metrics)]) over all the free space and each region,
    as a small table: one column per region, under a bar of its colour (as in the regions panel);
    the lower error of each column in bold, "–" where the region is empty."""
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cols = [("All", "nrmse", None)] + [(REGION_LABEL[k].replace(" ", "\n"), f"nrmse_{k}", REGION_COLOR[k])
                                       for k in REGION_NAMES]
    x0, dx = 0.19, 0.81 / len(cols)
    y_head, y_bar, y_rows = 0.84, 0.6, (0.4, 0.13)
    ax.text(0.0, y_head, "NRMSE (%)", ha="left", va="center", fontsize=7, style="italic")
    for c, (name, key, color) in enumerate(cols):
        xc = x0 + (c + 0.5) * dx
        ax.text(xc, y_head, name, ha="center", va="center", fontsize=7, linespacing=0.9)
        if color:
            ax.plot([xc - 0.38 * dx, xc + 0.38 * dx], [y_bar, y_bar], color=color, lw=2.5, solid_capstyle="butt")
        vals = [m.get(key, np.nan) for _, m in rows]
        best = None if all(np.isnan(vals)) else f"{100 * np.nanmin(vals):.1f}"
        for v, y in zip(vals, y_rows):
            text = "–" if np.isnan(v) else f"{100 * v:.1f}"
            ax.text(xc, y, text, ha="center", va="center", fontsize=7, fontweight="bold" if text == best else "normal")
    for (label, _), y in zip(rows, y_rows):
        ax.text(0.0, y, label, ha="left", va="center", fontsize=7)


def examples_figure(log, by_house, houses, run, out, shown=("LFD", "Gas Splatting"), copy_to=None):
    """run None: the median run of each house (median_run). Under the reconstructions, a table of
    the NRMSE of each method over all the free space and each region (region_table); its colour
    bars name the colours of the regions panel, so the figure needs no legend."""
    rows = []
    for h in houses:
        rr = run if run is not None else median_run(by_house[h])
        r = next(x for x in by_house[h] if x["run"] == rr)
        d = run_maps(log, r["case"], rr)
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
    table_h = 0.55                                                    # inches
    fig = plt.figure(figsize=(7.16, sum(map_w * q + 0.3 + table_h for q in ratios)))
    heights = []
    for q in ratios:
        heights += [q + 0.3 / map_w, table_h / map_w]
    grid = fig.add_gridspec(2 * len(rows), ncol, height_ratios=heights, wspace=0.05, hspace=0.08)
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
            ax = fig.add_subplot(grid[2 * i, j])
            if j == 1:
                img, kw = crop(x["regions"]).astype(float), dict(cmap=region_cmap, vmin=-0.5, vmax=3.5,
                                                                 interpolation="nearest")
                title = "Regions"
            else:
                img, kw = crop(x["gt"] if j == 0 else x["d"][f"pred/{shown[j - 2]}"]), dict(vmax=vmax)
                title = f"{x['name'].capitalize()}: ground truth" if j == 0 else LABEL.get(shown[j - 2], shown[j - 2])
            draw_gas_map(ax, img, obs, extent, title=f"({next(letters)}) {title}", fontsize=8.5, **kw)
            if j < 2:
                ax.scatter(P[inside, 0], P[inside, 1], marker="^", s=20, c="w", edgecolors="k", linewidths=0.6,
                           zorder=3)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        table = [(LABEL.get(m, m), gas_map_metrics(x["gt"], x["d"][f"pred/{m}"], x["d"]["obstacles"], 0.1, x["d"]["levels"]))
                 for m in shown]
        region_table(fig.add_subplot(grid[2 * i + 1, 2:]), table)
    save_figure(fig, out, copy_to=copy_to)


def main():
    ap = argparse.ArgumentParser(description="Paper figures of the comparison in the houses")
    ap.add_argument("--log", default=os.path.join(RESULTS_DIR, "random_poses_all.jsonl"))
    ap.add_argument("--examples", nargs=2, default=["house03", "house27"])
    ap.add_argument("--run", type=int, default=None,
                    help="Run shown in the examples (default: the median run of each house, GASPLAT minus LFD)")
    ap.add_argument("--output_dir", default=PLOTS_DIR)
    ap.add_argument("--copy_to", default=None, help="Also copy the PDFs here (e.g. ~/paper/figs)")
    args = ap.parse_args()
    by_house = load(args.log)
    methods = METHODS + [NO_PENALTY]
    outs = [os.path.join(args.output_dir, n) for n in ("houses_errors", "houses_metrics", "houses_examples")]
    errors_figure(by_house, methods, outs[0], args.copy_to)
    metrics_figure(by_house, methods, outs[1], args.copy_to)
    examples_figure(args.log, by_house, args.examples, args.run, outs[2], copy_to=args.copy_to)
    print(f"{len(by_house)} houses, runs per house: {sorted({len(v) for v in by_house.values()})}")


if __name__ == "__main__":
    main()
