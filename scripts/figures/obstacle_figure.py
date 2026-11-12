"""Figure: what the obstacle penalty prevents.

Two rooms share a wall without a door. The plume lies in the first room, against
the wall, and the robot only scans from that room, so every beam stops at the
wall and the second room is never observed. A primitive fitted to the plume can
then reach through the wall: the part of it inside the wall or beyond it changes
no measurement, so nothing in the data stops it. Without the penalty the gas
runs through the wall into the second room; with it, the primitives stay in free
space and the gas stops at the wall.

Walls are drawn as outlines and the field everywhere, so that the gas the model
places inside the wall is visible too. Primitives are drawn as their centers, as
in the injection figure (scripts/figures/injection_figure.py), whose style this
follows. Every panel comes from the real pipeline: beams from
generate_vantage_beams (the pan sweep of the experiments), measurements from the
same ray-cell integration, and the model trained by Trainer with and without the
penalty. Sized for one column.

    python scripts/figures/obstacle_figure.py [--copy_to ~/paper/figs]
"""
import os
import sys
import copy
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.sim_utils import (MeasurementBatch, EnvironmentContext, GroundTruth, generate_vantage_beams,
                             simulate_gas_integrals)
from utils.figure_style import GAS_CMAP_CLEAR, WALL_DARK, PRIMITIVE_GREY, apply_paper_style, save_figure

apply_paper_style("drawing", axes_titlesize=8.5)

COLUMN_WIDTH = 3.46                    # inches, one column of the paper (cas-dc)
MAP_SIZE = (5.0, 6.0)
CELL = 0.1
WALL = 0.3                             # thickness of every wall (m), as in the VGR houses
WALL_X = (3.0, 3.3)                    # the shared wall, full height, no door (m)
PLUME = ((3.0, 3.0), (0.8, 1.3), 1.0)  # center, (sigma_x, sigma_y), peak: highest right at the wall
POSES = np.array([[0.7, 0.7], [0.7, 5.3], [2.5, 0.7]])  # all in the first room
ANGULAR_STEP = 6.0

C_BEAM = "#9e9e9e"


def rooms():
    """Interior of both rooms, (x0, y0, x1, y1) in meters; everything else is wall."""
    map_w, map_h = MAP_SIZE
    return [(WALL, WALL, WALL_X[0], map_h - WALL), (WALL_X[1], WALL, map_w - WALL, map_h - WALL)]


def build_scene():
    """Ground truth and occupancy: outer walls, the shared wall, and the plume in the first room."""
    map_w, map_h = MAP_SIZE
    cols, rows = round(map_w / CELL), round(map_h / CELL)
    obstacles = np.ones((rows, cols), dtype=np.float32)
    for x0, y0, x1, y1 in rooms():
        obstacles[round(y0 / CELL):round(y1 / CELL), round(x0 / CELL):round(x1 / CELL)] = 0.0

    xs = (np.arange(cols) + 0.5) * CELL
    ys = (np.arange(rows) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    (mx, my), (sx, sy), rho = PLUME
    gas = rho * np.exp(-0.5 * (((gx - mx) / sx) ** 2 + ((gy - my) / sy) ** 2))
    # The gas is confined to the first room: none in the walls or beyond the shared one
    gas[(obstacles > 0.5) | (gx > WALL_X[0])] = 0.0
    return gas.astype(np.float32), obstacles


def fit(cfg: Config, batch: MeasurementBatch, environment: EnvironmentContext, obstacle_lambda: float):
    cfg = copy.deepcopy(cfg)
    cfg.train.obstacle_lambda = obstacle_lambda
    torch.manual_seed(0)
    model, _ = setup_gs_model(batch, cfg)
    trainer = Trainer(model, cfg, environment=environment)
    trainer.train(batch)
    trainer.finish()
    return model


def draw_base(ax, field, obstacles, vmax, title):
    """The field everywhere, walls as outlines only, so gas inside them shows."""
    map_w, map_h = MAP_SIZE
    ax.imshow(field, extent=(0, map_w, 0, map_h), origin="lower", cmap=GAS_CMAP_CLEAR, vmin=0, vmax=vmax,
              interpolation="bilinear", zorder=1)
    for x0, y0, x1, y1 in [(0.0, 0.0, map_w, map_h)] + rooms():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=WALL_DARK, lw=0.6, zorder=5))
    ax.text((WALL_X[1] + map_w) / 2, map_h / 2, "unobserved room", rotation=90, ha="center", va="center",
            fontsize=7.5, color="0.45", style="italic", zorder=6)
    ax.set_title(title, pad=2)
    ax.set_xlim(-0.05, map_w + 0.05)
    ax.set_ylim(-0.05, map_h + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Figure of what the obstacle penalty prevents")
    parser.add_argument("--copy_to", default=None, help="Also copy the PDF here (e.g. the paper's figs folder)")
    args = parser.parse_args()

    gas, obstacles = build_scene()
    beams, _, _ = generate_vantage_beams(obstacles > 0.5, CELL, ANGULAR_STEP, poses=POSES,
                                         rng=np.random.default_rng(0))
    y = simulate_gas_integrals(gas, beams, CELL, quiet=True)
    device = torch.device("cpu")
    batch = MeasurementBatch(beams=torch.tensor(beams, dtype=torch.float32, device=device),
                             measurements=torch.tensor(y, dtype=torch.float32, device=device))

    cfg = Config(env=EnvironmentParams(map_size=MAP_SIZE, cell_size=CELL), sim=SimulationParams(),
                 train=TrainParams(), densify=DensificationParams())
    cfg.device_type = "cpu"
    cfg.quiet = True
    environment = EnvironmentContext(obstacles=obstacles, ground_truth=GroundTruth(gas_map=gas, y_true=batch.measurements))

    lam = TrainParams().obstacle_lambda
    models = {l: fit(cfg, batch, environment, l) for l in (0.0, lam)}
    fields = {l: m.render_map(cell_size=CELL) for l, m in models.items()}
    vmax = max(gas.max(), *(f.max() for f in fields.values()))

    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_WIDTH, 1.75), constrained_layout=True)
    draw_base(axes[0], gas, obstacles, vmax, "(a) Ground truth")
    axes[0].add_collection(LineCollection(beams, colors=C_BEAM, lw=0.25, alpha=0.7, zorder=2))
    axes[0].plot(POSES[:, 0], POSES[:, 1], "s", color="k", ms=2.0, zorder=7)

    xs = (np.arange(gas.shape[1]) + 0.5) * CELL
    free = obstacles < 0.5
    room_b = free & (xs[None, :] > WALL_X[1])
    wall = (obstacles > 0.5) & (xs[None, :] >= WALL_X[0]) & (xs[None, :] <= WALL_X[1])
    stats = {}
    for ax, (l, title) in zip(axes[1:], ((0.0, "(b) Without penalty"), (lam, "(c) With penalty"))):
        draw_base(ax, fields[l], obstacles, vmax, title)
        with torch.no_grad():
            pos = models[l].get_pos().cpu().numpy()
        ax.plot(pos[:, 0], pos[:, 1], ".", color=PRIMITIVE_GREY, ms=1.5, zorder=6)
        f = fields[l]
        stats[l] = (models[l].num_gaussians, f[room_b].sum() / f.sum(), f[room_b].max() / gas.max(),
                    f[wall].sum() / f.sum(), np.sqrt(np.mean((f - gas)[free & ~room_b] ** 2)) / gas.max())

    fig.legend(handles=[
        Line2D([], [], color=C_BEAM, lw=1.0, label="beams"),
        Line2D([], [], color="k", marker="s", ls="", ms=2.0, label="emitter"),
        Line2D([], [], color=PRIMITIVE_GREY, marker=".", ls="", ms=4, label="primitive centers"),
    ], loc="outside lower center", ncol=3, frameon=False, handlelength=1.3, handletextpad=0.5,
        columnspacing=1.2, fontsize=8)

    save_figure(fig, "obstacle_penalty_toy", dpi={"png": 300}, copy_to=args.copy_to)

    print(f"beams: {len(beams)} from {len(POSES)} poses, none in the second room")
    for l, (k, share, peak, in_wall, err) in stats.items():
        print(f"lambda {l:g}: {k} primitives; unobserved room {100 * share:.1f}% of the gas (max {peak:.2f} x peak), "
              f"wall {100 * in_wall:.1f}%; NRMSE in the first room {100 * err:.2f}%")


if __name__ == "__main__":
    main()
