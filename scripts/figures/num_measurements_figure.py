"""Paper figures of the experiment on the number of measurements, from its results.

Quantitative figure (<output>.pdf): (a) the ground truth with every pose of the
experiment, numbered in the order the poses are added (fewer poses are always the
first ones), (b) NRMSE and (c) computation time of every method along the axis of
the experiment (poses, angular step or beams), mean and one standard deviation over
the sweep headings (over the seeds for procedural maps). The time of a grid method
includes the construction of its system matrix, so every time covers the whole
reconstruction from the measurements to the map.

Qualitative figure (<output>_maps.pdf, pose axis only): one row per number of poses
(--map_steps), with the ground truth and the beams, then the reconstruction of every
method from those measurements (sweep headings --seed), regenerated with the
configuration of the experiment. All maps share the colour scale of the ground truth.
Training is chaotic, so the reconstructions are kept next to the results
(<results>_maps_seed<seed>.npz) and reused, so that the figure does not change when it
is regenerated; --refresh computes them again.

Paper figures (side plume):
    python scripts/experiments/num_measurements.py --scenario data/scenarios/10x6_empty_room_side_plume_h0.75_t1000 \\
        --poses 2 3 4 6 8 12 16 --runs 5 --workers 1
    python scripts/figures/num_measurements_figure.py \\
        results/num_measurements_10x6_empty_room_side_plume_h0.75_t1000_poses_<date>.json \\
        --output plots/num_measurements_side_plume --copy_to ~/paper/figs
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import config_from_dict
from utils.gaden_utils import DEFAULT_ROOT
from utils.metrics import gas_map_metrics
from utils.scenario_sets import Case
from utils.experiment_utils import measurements_at, run_methods, total_time
from utils.figure_style import (GAS_CMAP, METHOD_ORDER, apply_paper_style, draw_gas_map, line_style, method_label,
                                save_figure)

apply_paper_style()

BEAM_COLOR = "0.55"    # beams of the ground-truth panel: light and faint, so that the plume stays visible
BEAM_ALPHA = 0.15
# Axis of the experiment, for results saved before the metadata had "axis"
AXES = {"poses": "poses", "beams": "beams", "angular step (deg)": "angular_step"}


def step_text(step: float, axis: str) -> str:
    return {"poses": f"{step:g} poses", "angular_step": f"{step:g}° step", "beams": f"{step:g} beams"}[axis]


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
            time.append([total_time(r) for r in rs])
        out[m] = {k: (np.mean(v, axis=1), np.std(v, axis=1)) for k, v in (("nrmse", nrmse), ("time", time))}
    beams = [np.mean([r["beams"] for r in runs if r["step"] == s]) for s in steps]
    return np.array(steps, dtype=float), np.array(beams), out


def pose_positions(batch) -> np.ndarray:
    """Common origin of the beams of each pose, in the order of the pose ids (the order they are added)."""
    beams, ids = batch.beams.cpu().numpy(), batch.pose_ids.cpu().numpy()
    return np.array([beams[ids == p][0, 0] for p in np.unique(ids)])


def quantitative_figure(data, case, cfg, axis, output, copy_to=None):
    meta = data["metadata"]
    steps = meta["steps"]
    procedural = case.scenario is None
    x_steps, n_beams, curve = curves(data, meta["cases"] if procedural else [case.name])
    x = x_steps if axis == "poses" else n_beams

    # Ground truth and every pose of the experiment (its largest step)
    batch, environment = measurements_at(cfg, max(steps) if axis == "poses" else steps[-1], axis)
    gt = environment.ground_truth.gas_map
    obstacles = environment.obstacles > 0.5 if environment.obstacles is not None else np.zeros(gt.shape, dtype=bool)
    H, W = gt.shape
    extent = (0, W * cfg.env.cell_size, 0, H * cfg.env.cell_size)

    fig = plt.figure(figsize=(7.16, 2.35))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.42, left=0.01, right=0.99, top=0.80, bottom=0.30)
    ax = fig.add_subplot(grid[0, 0])
    draw_gas_map(ax, gt, obstacles, extent, gt[~obstacles].max(), "(a) Ground truth and poses")
    if batch.pose_ids is not None:
        poses = pose_positions(batch)
        span = max(extent[1], extent[3])
        for i, p in enumerate(poses):
            # Each pose is a dot; its number sits on it, or is pushed away from a close neighbour
            # (with a leader line) so that the numbers do not overlap
            d = np.linalg.norm(poses - p, axis=1)
            d[i] = np.inf
            label_at = p
            if d.min() < 0.06 * span:
                away = p - poses[np.argmin(d)]
                label_at = p + 0.05 * span * away / max(np.linalg.norm(away), 1e-9)
                ax.plot(*zip(p, label_at), color="k", lw=0.5, zorder=3)
            ax.scatter(*p, s=5, c="k", zorder=4)
            ax.scatter(*label_at, s=62, c="w", edgecolors="k", linewidths=0.6, zorder=4)
            ax.text(*label_at, str(i + 1), ha="center", va="center", fontsize=6, zorder=5)
    sm = plt.cm.ScalarMappable(cmap=GAS_CMAP, norm=plt.Normalize(0, gt[~obstacles].max()))
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.06, pad=0.04, aspect=30)
    cbar.set_label("Concentration (ppm)", labelpad=1)
    cbar.ax.tick_params(labelsize=7.5)

    plots = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])]
    for ax, key, factor, ylabel in ((plots[0], "nrmse", 100, "NRMSE (% of peak)"),
                                    (plots[1], "time", 1, "Computation time (s)")):
        for m in METHOD_ORDER:
            if m not in curve:
                continue
            center, spread = factor * curve[m][key][0], factor * curve[m][key][1]
            ax.plot(x, center, **line_style(m))
            ax.fill_between(x, np.maximum(center - spread, 0), center + spread, color=line_style(m)["color"],
                            alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:.0f}" for v in x]); ax.minorticks_off()
        ax.set_xlabel("Number of poses" if axis == "poses" else "Number of beams")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.3)
        if axis != "beams":   # the other quantity on top
            top = ax.secondary_xaxis("top")
            top.set_xticks(x); top.minorticks_off()
            if axis == "poses":
                top.set_xticklabels([f"{b:.0f}" for b in n_beams], fontsize=7.5)
                top.set_xlabel("Number of beams", labelpad=2)
            else:
                top.set_xticklabels([f"{s:g}°" for s in steps], fontsize=7.5)
                top.set_xlabel("Angular step at each pose", labelpad=2)
    for ax, letter in zip(plots, "bc"):
        ax.text(-0.3, 1.3, f"({letter})", transform=ax.transAxes, ha="left", va="bottom")
    fig.legend(*plots[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.2, handlelength=2.2)
    save_figure(fig, output, copy_to=copy_to)


def reconstructions(case, cfg, axis, map_steps, cache, refresh=False):
    """Measurements and reconstructions of every method at each step of map_steps:
    [(step, batch, environment, metrics, predictions)]. The reconstructions are read
    from the cache (.npz) when it has them (unless refresh), else computed and saved."""
    saved = dict(np.load(cache)) if os.path.exists(cache) and not refresh else {}
    rows, new = [], False
    for step in map_steps:
        batch, environment = measurements_at(cfg, step, axis)
        keys = {m: f"{case.name}/{step:g}/{m}" for m in METHOD_ORDER}
        if all(k in saved for k in keys.values()):
            preds = {m: saved[k] for m, k in keys.items()}
            metrics = {m: gas_map_metrics(environment.ground_truth.gas_map, p, environment.obstacles,
                                          cfg.env.cell_size) for m, p in preds.items()}
        else:
            metrics, preds = run_methods(batch, environment, cfg, METHOD_ORDER)
            saved.update({k: np.asarray(preds[m]) for m, k in keys.items()})
            new = True
        rows.append((step, batch, environment, metrics, preds))
    if new:
        np.savez_compressed(cache, **saved)
        print(f"Reconstructions saved in {cache}")
    return rows


def qualitative_figure(rows, cfg, axis, output, copy_to=None):
    gt = rows[0][2].ground_truth.gas_map
    obstacles = rows[0][2].obstacles > 0.5 if rows[0][2].obstacles is not None else np.zeros(gt.shape, dtype=bool)
    H, W = gt.shape
    extent = (0, W * cfg.env.cell_size, 0, H * cfg.env.cell_size)
    vmax = gt[~obstacles].max()

    cols = ["truth"] + METHOD_ORDER
    panel_w = 7.16 / len(cols)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(7.16, len(rows) * panel_w * H / W * 1.08 + 0.75),
                             gridspec_kw=dict(wspace=0.04, hspace=0.06, left=0.035, right=0.995, top=0.92, bottom=0.17))
    for r, (step, batch, environment, metrics, preds) in enumerate(rows):
        for c, col in enumerate(cols):
            ax = axes[r, c]
            if col == "truth":
                draw_gas_map(ax, gt, obstacles, extent, vmax, "Ground truth" if r == 0 else "")
                ax.add_collection(LineCollection(batch.beams.cpu().numpy(), colors=BEAM_COLOR, linewidths=0.3,
                                                 alpha=0.3))
                pos = pose_positions(batch)
                ax.scatter(pos[:, 0], pos[:, 1], marker="^", s=16, c="w", edgecolors="k", linewidths=0.5, zorder=3)
                ax.set_ylabel(step_text(step, axis), fontsize=9)
                ax.yaxis.set_label_coords(-0.06, 0.5)
            else:
                draw_gas_map(ax, preds[col], obstacles, extent, vmax, method_label(col) if r == 0 else "")
                ax.text(0.98, 0.04, f"{100 * metrics[col]['nrmse']:.1f}%", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=7.5, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.8))
            for side in ax.spines.values():
                side.set_visible(True); side.set_color("0.75"); side.set_linewidth(0.5)
    sm = plt.cm.ScalarMappable(cmap=GAS_CMAP, norm=plt.Normalize(0, vmax))
    cax = fig.add_axes([0.3, 0.07, 0.4, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("Concentration (ppm); numbers: NRMSE", labelpad=1)
    cbar.ax.tick_params(labelsize=7.5)
    save_figure(fig, output, copy_to=copy_to)
    for step, _, _, metrics, _ in rows:
        print(f"{step_text(step, axis)}: " + ", ".join(f"{method_label(m)} {100 * metrics[m]['nrmse']:.2f}%"
                                                         for m in METHOD_ORDER))


def main():
    parser = argparse.ArgumentParser(description="Figures of the experiment on the number of measurements")
    parser.add_argument("results", help="Results of scripts/experiments/num_measurements.py (JSON)")
    parser.add_argument("--case", default=None, help="Case shown (default: the first of the experiment)")
    parser.add_argument("--map_steps", nargs="+", type=float, default=[2, 4, 8],
                        help="Steps of the experiment shown in the qualitative figure (pose axis)")
    parser.add_argument("--seed", type=int, default=0, help="Sweep headings of the reconstructions shown (GADEN)")
    parser.add_argument("--output", default=None, help="Output path without extension (default: plots/<experiment name>)")
    parser.add_argument("--copy_to", default=None, help="Also copy the PDFs here (e.g. the paper's figs folder)")
    parser.add_argument("--refresh", action="store_true",
                        help="Recompute the reconstructions of the qualitative figure instead of reading them from "
                             "<results>_maps_seed<seed>.npz")
    args = parser.parse_args()

    data = json.load(open(args.results))
    meta = data["metadata"]
    axis = meta.get("axis") or AXES[meta["step_unit"]]
    case = case_of(meta, args.case or meta["cases"][0])
    if case.name not in meta["cases"]:
        sys.exit(f"--case must be one of the cases of the experiment: {meta['cases']}")
    cfg = case.apply(config_from_dict(meta["config"]))
    cfg.quiet = True
    if case.scenario is not None:
        cfg.sim.seed = args.seed
    output = args.output or os.path.join("plots", meta["experiment_name"])

    quantitative_figure(data, case, cfg, axis, output, args.copy_to)
    if axis == "poses":
        cache = f"{os.path.splitext(args.results)[0]}_maps_seed{args.seed}.npz"
        rows = reconstructions(case, cfg, axis, args.map_steps, cache, args.refresh)
        qualitative_figure(rows, cfg, axis, output + "_maps", args.copy_to)
    print("results:", args.results)


if __name__ == "__main__":
    main()
