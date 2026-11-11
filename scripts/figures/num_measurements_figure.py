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

Paper figures (side plume):
    python scripts/experiments/num_measurements.py --scenario data/scenarios/10x6_empty_room_side_plume_h0.75_t1000 \\
        --poses 2 3 4 6 8 12 16 --repeats 5 --workers 1
    python scripts/figures/num_measurements_figure.py \\
        results/num_measurements_10x6_empty_room_side_plume_h0.75_t1000_poses_<date>.json \\
        --output plots/num_measurements_side_plume --copy_to ~/paper/figs
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


def style_kw(m: str) -> dict:
    style = AVAILABLE_METHODS[m]["style"]
    gs = m == "Gas Splatting"
    return dict(color=style["color"], marker=style["marker"], lw=1.8 if gs else 1.1, ms=7 if gs else 4.5, label=label(m))


def pose_positions(batch) -> np.ndarray:
    """Common origin of the beams of each pose, in the order of the pose ids (the order they are added)."""
    beams, ids = batch.beams.cpu().numpy(), batch.pose_ids.cpu().numpy()
    return np.array([beams[ids == p][0, 0] for p in np.unique(ids)])


def quantitative_figure(data, case, cfg, axis, output):
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
    draw_map(ax, gt, obstacles, extent, gt[~obstacles].max(), "(a) Ground truth and poses")
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
        for m in ORDER:
            if m not in curve:
                continue
            center, spread = factor * curve[m][key][0], factor * curve[m][key][1]
            ax.plot(x, center, **style_kw(m))
            ax.fill_between(x, np.maximum(center - spread, 0), center + spread, color=style_kw(m)["color"],
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
    save(fig, output)


def qualitative_figure(meta, case, cfg, axis, map_steps, output):
    rows = []
    for step in map_steps:
        batch, environment = measurements_at(cfg, step, axis)
        metrics, preds = run_methods(batch, environment, cfg, ORDER)
        rows.append((step, batch, environment, metrics, preds))
    gt = rows[0][2].ground_truth.gas_map
    obstacles = rows[0][2].obstacles > 0.5 if rows[0][2].obstacles is not None else np.zeros(gt.shape, dtype=bool)
    H, W = gt.shape
    extent = (0, W * cfg.env.cell_size, 0, H * cfg.env.cell_size)
    vmax = gt[~obstacles].max()

    cols = ["truth"] + ORDER
    panel_w = 7.16 / len(cols)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(7.16, len(rows) * panel_w * H / W * 1.08 + 0.75),
                             gridspec_kw=dict(wspace=0.04, hspace=0.06, left=0.035, right=0.995, top=0.92, bottom=0.17))
    for r, (step, batch, environment, metrics, preds) in enumerate(rows):
        for c, col in enumerate(cols):
            ax = axes[r, c]
            if col == "truth":
                draw_map(ax, gt, obstacles, extent, vmax, "Ground truth" if r == 0 else "")
                ax.add_collection(LineCollection(batch.beams.cpu().numpy(), colors=BEAM_COLOR, linewidths=0.3,
                                                 alpha=0.3))
                pos = pose_positions(batch)
                ax.scatter(pos[:, 0], pos[:, 1], marker="^", s=16, c="w", edgecolors="k", linewidths=0.5, zorder=3)
                ax.set_ylabel(step_text(step, axis), fontsize=9)
                ax.yaxis.set_label_coords(-0.06, 0.5)
            else:
                draw_map(ax, preds[col], obstacles, extent, vmax, label(col) if r == 0 else "")
                ax.text(0.98, 0.04, f"{100 * metrics[col]['nrmse']:.1f}%", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=7.5, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.8))
            for side in ax.spines.values():
                side.set_visible(True); side.set_color("0.75"); side.set_linewidth(0.5)
    sm = plt.cm.ScalarMappable(cmap=GAS_CMAP, norm=plt.Normalize(0, vmax))
    cax = fig.add_axes([0.3, 0.07, 0.4, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("Concentration (ppm); numbers: NRMSE", labelpad=1)
    cbar.ax.tick_params(labelsize=7.5)
    save(fig, output)
    for step, _, _, metrics, _ in rows:
        print(f"{step_text(step, axis)}: " + ", ".join(f"{label(m)} {100 * metrics[m]['nrmse']:.2f}%" for m in ORDER))


def save(fig, output):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{output}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}.{{pdf,png}}")


def main():
    parser = argparse.ArgumentParser(description="Figures of the experiment on the number of measurements")
    parser.add_argument("results", help="Results of scripts/experiments/num_measurements.py (JSON)")
    parser.add_argument("--case", default=None, help="Case shown (default: the first of the experiment)")
    parser.add_argument("--map_steps", nargs="+", type=float, default=[2, 4, 8],
                        help="Steps of the experiment shown in the qualitative figure (pose axis)")
    parser.add_argument("--seed", type=int, default=0, help="Sweep headings of the reconstructions shown (GADEN)")
    parser.add_argument("--output", default=None, help="Output path without extension (default: plots/<experiment name>)")
    parser.add_argument("--copy_to", default=None, help="Also copy the PDFs here (e.g. the paper's figs folder)")
    args = parser.parse_args()

    data = json.load(open(args.results))
    meta = data["metadata"]
    axis = meta.get("axis") or AXES[meta["step_unit"]]
    case = case_of(meta, args.case or meta["cases"][0])
    if case.name not in meta["cases"]:
        sys.exit(f"--case must be one of the cases of the experiment: {meta['cases']}")
    cfg = case.apply(experiment_config(meta))
    if case.scenario is not None:
        cfg.sim.seed = args.seed
    output = args.output or os.path.join("plots", meta["experiment_name"])

    quantitative_figure(data, case, cfg, axis, output)
    outputs = [output]
    if axis == "poses":
        qualitative_figure(meta, case, cfg, axis, args.map_steps, output + "_maps")
        outputs.append(output + "_maps")
    if args.copy_to:
        os.makedirs(os.path.expanduser(args.copy_to), exist_ok=True)
        for o in outputs:
            shutil.copy(f"{o}.pdf", os.path.expanduser(args.copy_to))
    print("results:", args.results)


if __name__ == "__main__":
    main()
