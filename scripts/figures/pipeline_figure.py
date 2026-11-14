"""Panels of the paper's training-pipeline figure (figs/incremental-pipeline.pdf).

The figure itself is assembled by hand from these panels; this script draws them from a
real GASPLAT run on a small two-room scene (6 x 4 m, a partition wall with a door at the
top), so that every panel shows what the pipeline actually computes:

    measured_integrals          the beams of the vantage poses, coloured by their reading
    predicted_integrals_init    the same beams as predicted by the initial Gaussians
    gaussians_init              the initial Gaussians (field and centers), on the final colour scale
    gaussians_init_own_scale    the same on their own colour scale
    field_final                 the reconstructed field
    field_final_centers         the same with the primitive centers
    occupancy_distance_field    the ESDF of the occupancy map (the obstacle penalty's input)
    ground_truth                the simulated gas field (not in the figure; for checking)

The gas is a plume in the GADEN style: Gaussian filaments released at a source in the
left room and carried along a curved path that leaves through the door and turns down
into the right room, spreading and diluting as they travel, with a slow random meander.
Each filament spreads with the walking distance through free space (shortest 8-neighbour
paths on the grid), so no gas leaks through the partition wall.

Each panel is saved as PDF, SVG and PNG with a transparent background in --output
(default plots/pipeline_panels); --copy_to also copies them there (e.g. ~/paper/pipeline_panels).

    python scripts/figures/pipeline_figure.py [--copy_to ~/paper/pipeline_panels]
"""
import os
import sys
import copy
import shutil
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.obstacle_utils import compute_esdf
from utils.sim_utils import MeasurementBatch, EnvironmentContext, GroundTruth, generate_vantage_beams, simulate_gas_integrals
from utils.figure_style import GAS_CMAP_CLEAR, WALL_DARK, PRIMITIVE_GREY, apply_paper_style, figure_path

apply_paper_style("drawing")

MAP = (6.0, 4.0)                    # map size (m)
CELL = 0.1                          # cell size (m)
WALL = 0.2                          # outer wall thickness (m)
INNER = (3.0, 3.2, 0.0, 2.6)        # partition wall: x from, x to, y from, y to (door above it)
POSES = np.array([[0.5, 0.5], [0.5, 3.5], [2.6, 0.5], [5.5, 0.5], [5.5, 3.5]])
ANGULAR_STEP = 10.0                 # degrees between the beams of a pose
HEADING_SEED = 4                    # random start heading of each pose's sweep; with 0 the run grows a thin,
                                    # very concentrated primitive against the left wall (a degenerate fit
                                    # to a few nearly parallel beams), 4 is free of them

# Plume: path of the airflow from the source, number of filaments and seed
PATH = np.array([(0.9, 0.8), (1.5, 1.6), (2.3, 2.6), (3.1, 3.2), (3.9, 3.0), (4.7, 2.3), (5.2, 1.4)])
FILAMENTS = 320
SEED = 3

BEAM = LinearSegmentedColormap.from_list("beam", ["#a8a8a8", "#e07070", "#c51b1b"])


def occupancy() -> np.ndarray:
    """Occupied cells (H, W): the outer walls and the partition."""
    W, H = round(MAP[0] / CELL), round(MAP[1] / CELL)
    obs = np.ones((H, W), bool)
    obs[round(WALL / CELL):round((MAP[1] - WALL) / CELL), round(WALL / CELL):round((MAP[0] - WALL) / CELL)] = False
    obs[round(INNER[2] / CELL):round(INNER[3] / CELL), round(INNER[0] / CELL):round(INNER[1] / CELL)] = True
    return obs


def free_space_distances(obs: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Shortest distances (m) through free cells from each source cell (flat indices) to every
    cell, moving between 8-neighbours; inf where unreachable."""
    H, W = obs.shape
    rows, cols, weights = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = 0, H - dy
        x0, x1 = max(0, -dx), W - max(0, dx)
        ya, xa = np.mgrid[y0:y1, x0:x1]
        yb, xb = ya + dy, xa + dx
        free = ~obs[ya, xa] & ~obs[yb, xb]
        rows.append((ya * W + xa)[free]); cols.append((yb * W + xb)[free])
        weights.append(np.full(int(free.sum()), CELL * np.hypot(dy, dx)))
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, H * W))
    return dijkstra(graph.tocsr(), directed=False, indices=sources)


def plume(obs: np.ndarray) -> np.ndarray:
    """Gas field (H, W), peak 1: filaments along PATH, denser near the source, spreading and
    diluting downstream, with a slow lateral meander."""
    H, W = obs.shape
    rng = np.random.default_rng(SEED)
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(PATH, axis=0), axis=1))]
    path, length = CubicSpline(t, PATH), t[-1]
    s = np.sort(rng.uniform(0, 1, FILAMENTS) ** 1.7) * length                   # arc length of each filament
    u = s / length                                                               # 0 at the source, 1 at the end
    tangent = path(s, 1) / np.linalg.norm(path(s, 1), axis=1, keepdims=True)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    meander = np.convolve(rng.normal(0, 1, FILAMENTS + 40), np.ones(40) / np.sqrt(40), "valid")[:FILAMENTS]
    offset = meander * 0.25 * u + rng.normal(0, 1, FILAMENTS) * (0.05 + 0.18 * u)
    centers = path(s) + normal * offset[:, None]
    sigma = 0.07 + 0.32 * u ** 1.3
    amplitude = np.exp(-2.6 * u) * rng.uniform(0.6, 1.0, FILAMENTS)
    cx = np.clip((centers[:, 0] / CELL).astype(int), 0, W - 1)
    cy = np.clip((centers[:, 1] / CELL).astype(int), 0, H - 1)
    inside = ~obs[cy, cx]                                                        # filaments pushed into a wall are dropped
    d = free_space_distances(obs, cy[inside] * W + cx[inside])
    gas = (amplitude[inside, None] * np.exp(-0.5 * (d / sigma[inside, None]) ** 2)).sum(0).reshape(H, W)
    gas[obs] = 0
    return (gas / gas.max()).astype(np.float32)


# ---------- drawing ----------

def walls(ax):
    """Walls as outlines: the outer boundary, and the boundary of the free space (both rooms and the door)."""
    W, H = MAP
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=WALL_DARK, lw=0.5, zorder=5))
    free = [(WALL, WALL), (INNER[0], WALL), (INNER[0], INNER[3]), (INNER[1], INNER[3]), (INNER[1], WALL),
            (W - WALL, WALL), (W - WALL, H - WALL), (WALL, H - WALL), (WALL, WALL)]
    ax.plot(*zip(*free), color=WALL_DARK, lw=0.5, zorder=5)


def field(ax, f, vmax):
    ax.imshow(f, extent=(0, MAP[0], 0, MAP[1]), origin="lower", cmap=GAS_CMAP_CLEAR, vmin=0, vmax=vmax,
              interpolation="bilinear", zorder=1)


def beams_colored(ax, beams, values, vmax):
    # Square root, so that beams with little gas already stand out from the empty ones (grey)
    t = np.sqrt(np.clip(np.asarray(values) / vmax, 0, 1))
    ax.add_collection(LineCollection(beams, colors=BEAM(t), lw=0.8, zorder=3))
    ax.plot(POSES[:, 0], POSES[:, 1], "s", color="k", ms=2, zorder=6)


def centers(ax, model):
    with torch.no_grad():
        p = model.get_pos().cpu().numpy()
    ax.plot(p[:, 0], p[:, 1], ".", color=PRIMITIVE_GREY, ms=2.2, zorder=6)


def save_panel(folder, name, draw, copy_to=None):
    fig = plt.figure(figsize=(1.5, 1.5 * MAP[1] / MAP[0]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-.05, MAP[0] + .05); ax.set_ylim(-.05, MAP[1] + .05); ax.set_aspect("equal"); ax.axis("off")
    draw(ax)
    for ext in ("pdf", "svg", "png"):
        path = os.path.join(folder, f"{name}.{ext}")
        fig.savefig(path, transparent=True, dpi=400)
        if copy_to:
            shutil.copy(path, copy_to)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Panels of the training-pipeline figure")
    parser.add_argument("--output", default="pipeline_panels", help="Folder of the panels (default plots/pipeline_panels)")
    parser.add_argument("--copy_to", default=None, help="Also copy the panels here (e.g. ~/paper/pipeline_panels)")
    args = parser.parse_args()
    folder = figure_path(args.output)
    os.makedirs(folder, exist_ok=True)
    copy_to = os.path.expanduser(args.copy_to) if args.copy_to else None
    if copy_to:
        os.makedirs(copy_to, exist_ok=True)

    torch.set_num_threads(4)
    obs = occupancy()
    gas = plume(obs)
    beams, _, _ = generate_vantage_beams(obs, CELL, ANGULAR_STEP, poses=POSES, rng=np.random.default_rng(HEADING_SEED))
    y = simulate_gas_integrals(gas, beams, CELL, quiet=True)
    batch = MeasurementBatch(beams=torch.tensor(beams, dtype=torch.float32), measurements=torch.tensor(y, dtype=torch.float32))
    cfg = Config(env=EnvironmentParams(map_size=MAP, cell_size=CELL), sim=SimulationParams(), train=TrainParams(),
                 densify=DensificationParams())
    cfg.device_type, cfg.quiet = "cpu", True
    env = EnvironmentContext(obstacles=obs.astype(np.float32), ground_truth=GroundTruth(gas_map=gas, y_true=batch.measurements))

    torch.manual_seed(0)
    model, _ = setup_gs_model(batch, cfg)
    init = copy.deepcopy(model)
    trainer = Trainer(model, cfg, environment=env)
    trainer.train(batch)
    trainer.finish()
    with torch.no_grad():
        y_init = init(batch.beams).numpy()
        f_init, f_final = init.render_map(CELL), model.render_map(CELL)
    vmax, ymax = max(gas.max(), f_final.max()), y.max()
    nrmse = np.sqrt(np.mean((f_final - gas)[~obs] ** 2)) / gas.max()

    panels = {
        "measured_integrals": lambda ax: (walls(ax), beams_colored(ax, beams, y, ymax)),
        "predicted_integrals_init": lambda ax: (walls(ax), beams_colored(ax, beams, y_init, ymax)),
        "gaussians_init": lambda ax: (walls(ax), field(ax, f_init, vmax), centers(ax, init)),
        "gaussians_init_own_scale": lambda ax: (walls(ax), field(ax, f_init, f_init.max()), centers(ax, init)),
        "field_final": lambda ax: (walls(ax), field(ax, f_final, vmax)),
        "field_final_centers": lambda ax: (walls(ax), field(ax, f_final, vmax), centers(ax, model)),
        "occupancy_distance_field": lambda ax: (ax.imshow(np.clip(compute_esdf(obs.astype(np.float32), CELL), 0, None),
                                                          extent=(0, MAP[0], 0, MAP[1]), origin="lower", cmap="Blues_r",
                                                          vmin=0, vmax=1.5, zorder=1), walls(ax)),
        "ground_truth": lambda ax: (walls(ax), field(ax, gas, vmax)),
    }
    for name, draw in panels.items():
        save_panel(folder, name, draw, copy_to)
    print(f"{len(beams)} beams, Gaussians {init.num_gaussians} -> {model.num_gaussians}, NRMSE {100 * nrmse:.1f}%")
    print("panels in", folder + (f" and {copy_to}" if copy_to else ""))


if __name__ == "__main__":
    main()
