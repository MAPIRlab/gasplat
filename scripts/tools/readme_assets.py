"""
Visuals of the README (not paper figures), each in a light and a dark variant (the README
picks one with <picture> and prefers-color-scheme), saved in assets/:

    python scripts/tools/readme_assets.py gif     # gasplat_training_{light,dark}.gif
    python scripts/tools/readme_assets.py room    # room_comparison_{light,dark}.png
    python scripts/tools/readme_assets.py house   # house_comparison_{light,dark}.png

gif: GASPLAT trained incrementally on a GADEN scenario (--scenario, default the side plume
of the 10x6 empty room), one vantage pose after the other (--poses, in the coverage planner's
order, --seed): the beams of each new pose fade in, then the estimate refines. Each ellipse is
the 1-sigma contour of a primitive holding at least 5% of the largest concentration, with
capped opacity, as in the live visualizer. Needs ffmpeg (bundled with imageio-ffmpeg).

room: every method on the side plume with 2, 4 and 8 poses (seed 0), from the results of
scripts/experiments/num_measurements.py (--results, default the newest side-plume run). The
reconstructions are read from the cache of scripts/figures/num_measurements_figure.py, or
computed and cached when it has none.

house: one run of scripts/experiments/random_poses.py saved with --keep_maps (--maps).

Colour map: viridis, as in the plots of main.py, quick_compare.py and the live view. Text
font: Lato if installed, else DejaVu Sans.
"""
import os
import sys
import glob
import json
import argparse
import multiprocessing
import tempfile
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "scripts", "figures"))
from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams, config_from_dict
from utils.sim_utils import generate_simulation_data, MeasurementBatch
from utils.init_utils import setup_gs_model
from utils.metrics import gas_map_metrics
from utils.paths import RESULTS_DIR, SCENARIOS_DIR
from trainer import Trainer

CMAP = plt.get_cmap("viridis")
ASSETS = os.path.join(REPO, "assets")
ROOM = os.path.join(SCENARIOS_DIR, "10x6_empty_room_side_plume_h0.75_t1000")
ROOM_RESULTS = os.path.join(RESULTS_DIR, "num_measurements_10x6_empty_room_side_plume_h0.75_t1000_poses_*.json")
HOUSE_MAPS = os.path.join(RESULTS_DIR, "random_poses_all_maps", "house13_2,3-4,6_slow_h0.85_it1999_run0.npz")

# GitHub page colours (Primer): background, text, muted text, panel surface, accent, wall
THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", surface="#f6f8fa", accent="#0969da",
                  wall="#c8ccd1", chip="#ffffff", track="#d1d9e0"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", surface="#161b22", accent="#4493f8",
                 wall="#3d444d", chip="#0d1117", track="#3d444d"),
}
STYLE = {"font.family": ["Lato", "DejaVu Sans"], "font.size": 11, "mathtext.fontset": "dejavusans"}
plt.rcParams.update(STYLE)


# ---------- drawing helpers ----------

def rounded(ax, th, radius=0.035):
    """Rounded clip box of the axes (in axes coordinates); returns it to clip the artists."""
    box = FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={radius}", transform=ax.transAxes,
                         facecolor="none", edgecolor="none")
    ax.add_patch(box)
    return box


def map_axes(ax, th):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(th["bg"])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_map(ax, img, obstacles, extent, vmax, th, clip, empty=False):
    """Gas map (or an empty surface) with the walls, clipped to the rounded box."""
    if empty:
        base = ax.imshow(np.ones_like(img), origin="lower", extent=extent,
                         cmap=matplotlib.colors.ListedColormap([th["surface"]]), vmin=0, vmax=1)
    else:
        base = ax.imshow(np.where(obstacles, np.nan, img), origin="lower", extent=extent, cmap=CMAP, vmin=0,
                         vmax=vmax, interpolation="bilinear")
    walls = ax.imshow(np.where(obstacles, 1.0, np.nan), origin="lower", extent=extent,
                      cmap=matplotlib.colors.ListedColormap([th["wall"]]), interpolation="nearest")
    for a in (base, walls):
        a.set_clip_path(clip)
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])


def label(ax, text, th, size=12.5):
    ax.text(0.0, 1.035, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=size, color=th["fg"],
            fontweight="semibold")


def chip(ax, text, th, size=10):
    ax.text(0.975, 0.05, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=size, color=th["fg"],
            bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.5", facecolor=th["chip"], edgecolor="none", alpha=0.85))


def poses_scatter(ax, poses, th, size=46, highlight=None):
    ax.scatter(poses[:, 0], poses[:, 1], marker="^", s=size, c="white", edgecolors="#1f2328", linewidths=0.8, zorder=4)
    if highlight is not None:
        ax.scatter([highlight[0]], [highlight[1]], marker="^", s=size * 1.9, facecolors="none",
                   edgecolors=th["accent"], linewidths=1.6, zorder=5)


def ellipse_outlines(pos, cov, rho, max_alpha=0.22, points=48):
    """1-sigma outlines (K, T, 2) of the primitives holding >= 5% of the largest concentration, and
    their opacity, growing with the concentration up to max_alpha."""
    if not len(rho):
        return np.zeros((0, points, 2)), np.zeros(0)
    rel = rho / max(float(rho.max()), 1e-12)
    keep = rel >= 0.05
    w, v = np.linalg.eigh(cov[keep])                                  # cov = V diag(w) V^T
    t = np.linspace(0, 2 * np.pi, points)
    circle = np.stack([np.cos(t), np.sin(t)])                         # (2, T)
    curves = pos[keep][:, :, None] + (v * np.sqrt(np.maximum(w, 0))[:, None, :]) @ circle   # (K, 2, T)
    return curves.transpose(0, 2, 1), 0.06 + (max_alpha - 0.06) * np.minimum(1.0, rel[keep])


def ellipses(ax, pos, cov, rho, clip, fade=1.0):
    """The outlines in white, as one line collection (as separate patches, drawing them takes most
    of the frame time); fade scales their opacity (crossfades)."""
    curves, alpha = ellipse_outlines(pos, cov, rho)
    if not len(alpha):
        return
    colors = np.ones((len(alpha), 4))
    colors[:, 3] = alpha * fade
    lc = LineCollection(curves, colors=colors, linewidths=0.6)
    lc.set_clip_path(clip)
    ax.add_collection(lc)


def save_png(fig, path, th):
    fig.savefig(path, dpi=fig.dpi, facecolor=th["bg"])
    plt.close(fig)
    print(path)


# ---------- GIF ----------

class Snapshots:
    """Keeps the map and the primitives every `every` optimization steps of each batch (counted from
    its start, the GIF's frames) and right after every densification (as the live visualizer)."""
    def __init__(self, cell, every=25):
        self.cell, self.every = cell, every
        self.items, self.stage, self.stage_start = [], 0, 0

    def start_stage(self, stage, iteration):
        self.stage, self.stage_start = stage, iteration

    def step(self, iteration, model):
        """After every optimization step (wraps Trainer.optimization_step)."""
        if (iteration - self.stage_start) % self.every == 0:
            self.grab(iteration, model)

    def grab(self, iteration, model):
        with torch.no_grad():
            # Copies: on the CPU .numpy() is a view of the tensor, and the optimizer updates the
            # positions in place, so without them every snapshot would show the latest positions
            self.items.append(dict(it=float(iteration), stage=self.stage, map=model.render_map(cell_size=self.cell),
                                   pos=model.get_pos().cpu().numpy().copy(), cov=model.get_covariance().cpu().numpy().copy(),
                                   rho=model.get_concentration().cpu().numpy().copy()))

    def update(self, iteration, loss_history, model, beams=None):
        pass                                                  # the trainer's visualizer hook: unused

    def save_gif(self, path):
        pass


def train_incremental(scenario, num_poses, seed):
    torch.set_num_threads(4)
    cfg = Config(EnvironmentParams(), SimulationParams(), TrainParams(), DensificationParams())
    cfg.quiet, cfg.sim.scenario, cfg.sim.num_poses, cfg.sim.seed = True, scenario, num_poses, seed
    batch, env = generate_simulation_data(cfg)
    snaps = Snapshots(cfg.env.cell_size)
    model = trainer = None
    for p in range(num_poses):
        m = batch.pose_ids == p
        b = MeasurementBatch(batch.beams[m], batch.measurements[m], batch.pose_ids[m])
        if model is None:
            model, _ = setup_gs_model(b, cfg)
            trainer = Trainer(model, cfg, environment=env)
            trainer.visualizer = snaps
            step = trainer.optimization_step

            densify = trainer._densify

            def step_and_grab(iteration, step=step, trainer=trainer, model=model):
                out = step(iteration)
                snaps.step(trainer.global_iteration, model)
                return out

            def densify_and_grab(densify=densify, trainer=trainer, model=model):
                densify()
                snaps.grab(trainer.global_iteration + 0.5, model)       # between this step and the next
            trainer.optimization_step = step_and_grab
            trainer._densify = densify_and_grab
        snaps.start_stage(p, trainer.global_iteration)
        trainer.train(b)
        snaps.grab(trainer.global_iteration, model)          # the state at the end of the stage
    gt, obs = env.ground_truth.gas_map, env.obstacles > 0.5
    for s in snaps.items:
        s["nrmse"] = gas_map_metrics(gt, s["map"], obs)["nrmse"]
    beams, ids = batch.beams.cpu().numpy(), batch.pose_ids.cpu().numpy()
    return dict(gt=gt, obs=obs, cell=cfg.env.cell_size, beams=beams, ids=ids, snaps=snaps.items,
                poses=np.array([beams[ids == p][0, 0] for p in range(num_poses)]), num_poses=num_poses)


def frame_plan(run, fade=6):
    """Frames as (snapshot a, snapshot b, blend t, stage, fade of the newest beams, blank estimate):
    the state shown is snapshot a crossfaded with b by t. Each stage: the beams of the new pose fade
    in over the last state, then one frame every Snapshots.every iterations, the same time scale in
    every batch. A densification replaces many primitives at once (the last ones of a batch the most,
    as splitting doubles them), so its frame shows the states before and after it half and half,
    which splits the change over two frames without changing the timing."""
    snaps, plan, last = run["snaps"], [], None
    for p in range(run["num_poses"]):
        idx = [i for i, s in enumerate(snaps) if s["stage"] == p]
        for k in range(fade):
            i = last if last is not None else idx[0]
            plan.append((i, i, 0.0, p, (k + 1) / fade, last is None))
        for n, i in enumerate(idx):
            if snaps[i]["it"] % 1:                                    # right after a densification: shown
                continue                                              # blended into the frame before it
            nxt = idx[n + 1] if n + 1 < len(idx) else None
            if nxt is not None and snaps[nxt]["it"] == snaps[i]["it"] + 0.5:
                plan.append((i, nxt, 0.5, p, 1.0, False))
            else:
                plan.append((i, i, 0.0, p, 1.0, False))
        last = idx[-1]
    return plan


def gif_frame(run, plan_item, th, path):
    i, i2, t, stage, fade, blank = plan_item
    s, s2 = run["snaps"][i], run["snaps"][i2]
    shown_map = (1 - t) * s["map"] + t * s2["map"]
    it_now, nrmse_now = (1 - t) * s["it"] + t * s2["it"], (1 - t) * s["nrmse"] + t * s2["nrmse"]
    gt, obs, cell = run["gt"], run["obs"], run["cell"]
    H, W = gt.shape
    extent = (0, W * cell, 0, H * cell)
    vmax = gt[~obs].max()
    fig = plt.figure(figsize=(10.8, 3.4), dpi=100, facecolor=th["bg"])
    gs = fig.add_gridspec(2, 3, height_ratios=[1.98, 0.42], left=0.025, right=0.975, top=0.9, bottom=0.03,
                          wspace=0.06, hspace=0.22)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    for ax in axes:
        map_axes(ax, th)

    # Measurements: beams of the poses visited so far, the newest fading in in the accent colour
    ax = axes[0]
    clip = rounded(ax, th)
    draw_map(ax, gt, obs, extent, vmax, th, clip, empty=True)
    old = run["beams"][run["ids"] < stage]
    new = run["beams"][run["ids"] == stage]
    if len(old):
        lc = LineCollection(old, colors=th["accent"], linewidths=0.5, alpha=0.28); lc.set_clip_path(clip)
        ax.add_collection(lc)
    lc = LineCollection(new, colors=th["accent"], linewidths=0.7, alpha=0.75 * fade); lc.set_clip_path(clip)
    ax.add_collection(lc)
    poses_scatter(ax, run["poses"][:stage + 1], th, highlight=run["poses"][stage])
    label(ax, "Robot measurements", th)

    # Estimate with the primitives
    ax = axes[1]
    clip = rounded(ax, th)
    if blank:
        draw_map(ax, np.zeros_like(gt), obs, extent, vmax, th, clip)
    else:
        draw_map(ax, shown_map, obs, extent, vmax, th, clip)
        ellipses(ax, s["pos"], s["cov"], s["rho"], clip, fade=1 - t)
        if t > 0:
            ellipses(ax, s2["pos"], s2["cov"], s2["rho"], clip, fade=t)
    label(ax, "GASPLAT estimate", th)
    if not blank:
        chip(ax, f"{len((s2 if t >= 0.5 else s)['rho'])} Gaussians", th)

    ax = axes[2]
    clip = rounded(ax, th)
    draw_map(ax, gt, obs, extent, vmax, th, clip)
    label(ax, "Ground truth", th)

    # Bottom strip: poses visited | NRMSE over the iterations
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 0.5, f"Pose {stage + 1} of {run['num_poses']}", color=th["fg"], fontsize=12, va="center",
            fontweight="semibold")
    for k in range(run["num_poses"]):
        ax.scatter(0.42 + 0.075 * k, 0.5, s=60, marker="o", zorder=3,
                   c=th["accent"] if k <= stage else th["track"], edgecolors="none")

    ax = fig.add_subplot(gs[1, 1:]); ax.set_facecolor(th["bg"])
    total = run["snaps"][-1]["it"]
    curve = [(x["it"], 100 * x["nrmse"]) for x in run["snaps"][:i + 1]] + [(it_now, 100 * nrmse_now)]
    ax.plot([0, total], [0, 0], color=th["track"], lw=1)
    if not blank:
        cx, cy = zip(*curve)
        ax.fill_between(cx, cy, color=th["accent"], alpha=0.15, lw=0)
        ax.plot(cx, cy, color=th["accent"], lw=1.6)
        ax.text(1.0, 0.55, f"NRMSE {100 * nrmse_now:.1f}%", transform=ax.transAxes, ha="right", va="center",
                color=th["fg"], fontsize=12, fontweight="semibold")
    top = max(100 * x["nrmse"] for x in run["snaps"])
    ax.set_xlim(0, total * 1.22); ax.set_ylim(-0.04 * top, top * 1.05)
    ax.axis("off")
    fig.savefig(path, dpi=100, facecolor=th["bg"])
    plt.close(fig)


_RUN = None          # the trained run, shared with the frame workers (fork)


def _render(job):
    theme, item, path = job
    gif_frame(_RUN, item, THEMES[theme], path)


def make_gif(args):
    import imageio_ffmpeg
    global _RUN
    _RUN = run = train_incremental(args.scenario, args.poses, args.seed)
    plan = frame_plan(run)
    hold = 40                                                 # frames that hold the result (2 s)
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(name, item, os.path.join(tmp, f"{name}_{k:04d}.png"))
                for name in THEMES for k, item in enumerate(plan)]
        with multiprocessing.get_context("fork").Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
            pool.map(_render, jobs, chunksize=4)
        for name in THEMES:
            last = os.path.join(tmp, f"{name}_{len(plan) - 1:04d}.png")
            for k in range(len(plan), len(plan) + hold):
                os.link(last, os.path.join(tmp, f"{name}_{k:04d}.png"))
            out = os.path.join(ASSETS, f"gasplat_training_{name}.gif")
            # Ordered (Bayer) dithering: a colour gets the same pattern in every frame, so the static
            # parts stay still (error diffusion reshuffles its noise in every frame, which flickers)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "20",
                            "-i", os.path.join(tmp, f"{name}_%04d.png"), "-filter_complex",
                            "scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
                            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                            "-loop", "0", out], check=True)
            print(out, f"{os.path.getsize(out) / 1e6:.1f} MB", len(plan) + hold, "frames")
    print("NRMSE per stage:", [f"{100 * [s for s in run['snaps'] if s['stage'] == p][-1]['nrmse']:.1f}"
                               for p in range(run["num_poses"])])


# ---------- static comparisons ----------

def make_room(args, steps=(2, 4, 8), methods=("SART", "RBF Coupled SART", "LFD", "LTD", "Gas Splatting")):
    from num_measurements_figure import case_of, reconstructions
    plt.rcdefaults(); plt.rcParams.update(STYLE)       # the import applies the paper style
    found = sorted(glob.glob(args.results))
    if not found:
        sys.exit(f"No results match {args.results}: run scripts/experiments/num_measurements.py first (see the README)")
    results = found[-1]
    names = {"RBF Coupled SART": "RBF-SART", "Gas Splatting": "GASPLAT"}
    meta = json.load(open(results))["metadata"]
    case = case_of(meta, meta["cases"][0])
    cfg = case.apply(config_from_dict(meta["config"])); cfg.quiet, cfg.sim.seed = True, 0
    rows = reconstructions(case, cfg, "poses", steps, os.path.splitext(results)[0] + "_maps_seed0.npz")
    env, cell = rows[0][2], cfg.env.cell_size
    gt, obs = env.ground_truth.gas_map, env.obstacles > 0.5
    H, W = gt.shape
    extent = (0, W * cell, 0, H * cell)
    vmax = gt[~obs].max()
    cols = ["truth"] + list(methods)
    for name, th in THEMES.items():
        pw = 2.25
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(pw * len(cols) + 0.5, len(rows) * pw * H / W * 1.18 + 0.5),
                                 dpi=110, facecolor=th["bg"],
                                 gridspec_kw=dict(wspace=0.05, hspace=0.16, left=0.04, right=0.995, top=0.91,
                                                  bottom=0.02))
        for r, (step, batch, _, metrics, preds) in enumerate(rows):
            beams, ids = batch.beams.cpu().numpy(), batch.pose_ids.cpu().numpy()
            poses = np.array([beams[ids == p][0, 0] for p in np.unique(ids)])
            for c, col in enumerate(cols):
                ax = axes[r, c]; map_axes(ax, th); clip = rounded(ax, th, 0.05)
                if col == "truth":
                    draw_map(ax, gt, obs, extent, vmax, th, clip)
                    lc = LineCollection(beams, colors="white", linewidths=0.35, alpha=0.3); lc.set_clip_path(clip)
                    ax.add_collection(lc)
                    poses_scatter(ax, poses, th, size=26)
                    ax.text(-0.06, 0.5, f"{step:g} poses", transform=ax.transAxes, rotation=90, ha="center",
                            va="center", color=th["fg"], fontsize=11, fontweight="semibold")
                    if r == 0:
                        label(ax, "Ground truth", th, 12)
                else:
                    draw_map(ax, preds[col], obs, extent, vmax, th, clip)
                    chip(ax, f"{100 * metrics[col]['nrmse']:.1f}%", th, 9)
                    if r == 0:
                        label(ax, names.get(col, col), th, 12)
        save_png(fig, os.path.join(ASSETS, f"room_comparison_{name}.png"), th)


def make_house(args, methods=(("Gas Splatting", "GASPLAT"), ("LFD", "LFD"), ("LTD", "LTD"))):
    if not os.path.exists(args.maps):
        sys.exit(f"{args.maps} not found: run scripts/experiments/random_poses.py with --keep_maps first")
    d = np.load(args.maps)
    gt, obs, poses = d["gt"], d["obstacles"] > 0.5, d["poses"]
    H, W = gt.shape
    extent = (0, W * 0.1, 0, H * 0.1)
    vmax = gt[~obs].max()
    for name, th in THEMES.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 2 * 6 * H / W * 1.17 + 0.3), dpi=90, facecolor=th["bg"],
                                 gridspec_kw=dict(wspace=0.04, hspace=0.2, left=0.005, right=0.995, top=0.92,
                                                  bottom=0.01))
        ax = axes[0, 0]; map_axes(ax, th); clip = rounded(ax, th, 0.03)
        draw_map(ax, gt, obs, extent, vmax, th, clip)
        poses_scatter(ax, poses, th, size=34)
        label(ax, "Ground truth and robot poses", th, 13)
        for ax, (key, title) in zip(axes.flat[1:], methods):
            map_axes(ax, th); clip = rounded(ax, th, 0.03)
            pred = d[f"pred/{key}"]
            m = gas_map_metrics(gt, pred, obs)
            draw_map(ax, pred, obs, extent, vmax, th, clip)
            label(ax, title, th, 13)
            chip(ax, f"NRMSE {100 * m['nrmse']:.1f}%  ·  peak {100 * m['peak_ratio']:.0f}% of true", th, 10.5)
        save_png(fig, os.path.join(ASSETS, f"house_comparison_{name}.png"), th)


def main():
    parser = argparse.ArgumentParser(description="Visuals of the README, in light and dark variants")
    parser.add_argument("asset", choices=["gif", "room", "house"], help="Visual to generate")
    parser.add_argument("--scenario", default=ROOM, help="gif: GADEN scenario package")
    parser.add_argument("--poses", type=int, default=6, help="gif: vantage poses, added one at a time")
    parser.add_argument("--seed", type=int, default=0, help="gif: seed of the sweep headings")
    parser.add_argument("--results", default=ROOM_RESULTS,
                        help="room: results of num_measurements.py on the side plume (glob; the newest match is used)")
    parser.add_argument("--maps", default=HOUSE_MAPS, help="house: maps of one run saved by random_poses.py --keep_maps")
    args = parser.parse_args()
    {"gif": make_gif, "room": make_room, "house": make_house}[args.asset](args)


if __name__ == "__main__":
    main()
