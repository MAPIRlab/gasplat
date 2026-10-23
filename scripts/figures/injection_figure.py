"""Figure: the Gaussian injection mechanism, step by step.

The scenario is the one the mechanism exists for. A robot inspects a two-room
environment: from the first room it observes a plume and the model is fitted to
it, and it then moves to the second room, where a second plume has never been
observed. The arriving batch therefore crosses a region holding no primitives at
all, where the data gradient vanishes and the optimizer has nothing to act upon.

Every panel is produced by the real pipeline: the model is trained with Trainer,
the measurements come from the same ray-cell integration used in the experiments,
and the stages drawn below are checked against extract_candidate_positions.
"""
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.densification_utils import extract_candidate_positions
from utils.obstacle_utils import add_rectangle, obstacle_mask_to_geometry
from utils.sim_utils import MeasurementBatch, EnvironmentContext, GroundTruth, simulate_gas_integrals

# STIX (Times-like), shipped with matplotlib: heavier strokes than Computer Modern at the printed size
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 9,
})

MAP_SIZE = (12.0, 7.0)
CELL = 0.25

# Plume observed from the first room, and the one only the second batch reaches
PLUME_SEEN = ((3.0, 3.6), 1.10, 1.0)
PLUME_NEW = ((8.9, 4.9), 0.95, 1.2)

GAS_CMAP = LinearSegmentedColormap.from_list(
    "gas", [(1, 1, 1, 0), (1, 0.96, 0.35, 1), (0.93, 0.85, 0.10, 1)])
RES_CMAP = plt.get_cmap("OrRd")
C_MODEL = "#7a7a7a"      # primitives the model already holds
C_CAND = "#c51b1b"       # beams selected as candidates
C_NEW = "#2166ac"        # everything the injection produces
C_OFF = "#d9d9d9"        # beams taking no part in the step


# --------------------------------------------------------------------- scene
def build_scene():
    """Ground truth, walls and the two measurement batches of the inspection."""
    map_w, map_h = MAP_SIZE
    cols, rows = int(map_w / CELL), int(map_h / CELL)

    xs = (np.arange(cols) + 0.5) * CELL
    ys = (np.arange(rows) + 0.5) * CELL
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)

    gas = np.zeros((rows, cols), dtype=np.float32)
    for (mu, sigma, rho) in (PLUME_SEEN, PLUME_NEW):
        gas += rho * np.exp(-0.5 * np.sum((grid - np.asarray(mu)) ** 2, axis=-1) / sigma ** 2)

    # A partition wall with a doorway at the top, so the first room is observed
    # on its own while a few beams of the second batch still reach back into it
    obstacles = np.zeros((rows, cols), dtype=np.float32)
    add_rectangle(obstacles, 0.485, 0.22, 0.515, 1.00)
    geometry = obstacle_mask_to_geometry(obstacles, CELL)

    return gas, obstacles, geometry


def fan(origin, a0_deg, a1_deg, count, geometry):
    """A pan-tilt sweep from one vantage point, clipped at the walls it hits."""
    map_w, map_h = MAP_SIZE
    origin = np.asarray(origin, dtype=float)
    beams = []

    for angle in np.deg2rad(np.linspace(a0_deg, a1_deg, count)):
        d = np.array([np.cos(angle), np.sin(angle)])

        # Exit point through the boundary of the domain
        ts = []
        for lo, hi, comp in ((0.0, map_w, 0), (0.0, map_h, 1)):
            if abs(d[comp]) > 1e-9:
                for bound in (lo, hi):
                    t = (bound - origin[comp]) / d[comp]
                    p = origin + t * d
                    if t > 1e-6 and -1e-6 <= p[0] <= map_w + 1e-6 and -1e-6 <= p[1] <= map_h + 1e-6:
                        ts.append(t)
        if not ts:
            continue
        end = origin + min(ts) * d

        # Reflector sits on the first surface the beam meets
        if geometry is not None:
            line = LineString([tuple(origin), tuple(end)])
            hit = line.intersection(geometry)
            if not hit.is_empty:
                start = Point(*origin)
                dists = [start.distance(Point(c)) for g in getattr(hit, "geoms", [hit])
                         for c in np.atleast_2d(np.array(g.coords))]
                dists = [dd for dd in dists if dd > 1e-6]
                if dists:
                    end = origin + min(min(dists), min(ts)) * d

        end = np.clip(end, [0.0, 0.0], [map_w, map_h])
        beams.append((tuple(origin), tuple(end)))
    return beams


def make_batch(beams, gas, device):
    y = simulate_gas_integrals(gas, beams, CELL, quiet=True)
    return MeasurementBatch(
        beams=torch.tensor(beams, dtype=torch.float32, device=device),
        measurements=torch.tensor(y, dtype=torch.float32, device=device),
    )


# ---------------------------------------------------------------- mechanism
def injection_stages(beams, residuals, predictions, min_dist, existing_pos):
    """Reproduces extract_candidate_positions, keeping every intermediate for the figure."""
    admitted = residuals > predictions

    lengths = torch.clamp(torch.norm(beams[:, 1] - beams[:, 0], dim=1), min=1e-5)
    score = residuals / lengths
    admitted_scores = score[admitted]
    threshold = admitted_scores.mean() + admitted_scores.std()
    candidate = admitted & (score >= threshold)
    cand_idx = torch.where(candidate)[0]

    pts, pts_score = [], []
    cb, cs = beams[cand_idx], score[cand_idx]
    for a in range(len(cand_idx)):
        for b in range(a + 1, len(cand_idx)):
            (x1, y1), (x2, y2) = cb[a]
            (x3, y3), (x4, y4) = cb[b]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) <= 1e-7:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / (den + 1e-9)
            u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / (den + 1e-9)
            if 0.01 <= t <= 0.99 and 0.01 <= u <= 0.99:
                pts.append([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
                pts_score.append((cs[a] + cs[b]).item())

    pts = torch.tensor(pts, dtype=torch.float32).reshape(-1, 2)
    pts_score = torch.tensor(pts_score, dtype=torch.float32)
    if pts.shape[0] == 0:
        raise RuntimeError(
            f"no intersections: {int(admitted.sum())} admitted, {len(cand_idx)} candidate beams. "
            "The candidate beams have to come from more than one vantage point.")
    near = torch.cdist(pts, existing_pos.cpu()).min(dim=1).values < min_dist

    kept = []
    for i in torch.argsort(pts_score, descending=True).tolist():
        if near[i] or any(torch.norm(pts[k] - pts[i]) < min_dist for k in kept):
            continue
        kept.append(i)

    return dict(score=score, admitted=admitted, threshold=threshold,
                candidate=candidate, points=pts, point_score=pts_score, near=near,
                kept=torch.tensor(kept, dtype=torch.long))


# ------------------------------------------------------------------- drawing
def model_coverage(ax, model, level):
    """What the model currently explains: a contour of its field, plus its centers.

    Drawing the field rather than one ellipse per primitive keeps the panels
    readable, and it is the quantity the gate actually compares against.
    """
    map_w, map_h = MAP_SIZE
    field = model.render_map(cell_size=CELL)
    xs = (np.arange(field.shape[1]) + 0.5) * CELL
    ys = (np.arange(field.shape[0]) + 0.5) * CELL
    ax.contour(xs, ys, field, levels=[level], colors=[C_MODEL],
               linewidths=0.7, linestyles="--", zorder=4)
    with torch.no_grad():
        pos = model.get_pos().cpu().numpy()
    ax.plot(pos[:, 0], pos[:, 1], ".", color=C_MODEL, ms=1.2, zorder=5)


def exclusion_zone(ax, existing_pos, min_dist):
    """The region the proximity filter closes off, within min_dist of any primitive."""
    map_w, map_h = MAP_SIZE
    step = 0.05
    xs = np.arange(0, map_w + step, step)
    ys = np.arange(0, map_h + step, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    d = np.linalg.norm(pts[:, None, :] - existing_pos[None, :, :], axis=-1).min(axis=1)
    inside = (d <= min_dist).reshape(gy.shape)
    ax.contourf(xs, ys, inside.astype(float), levels=[0.5, 1.5],
                colors=[C_NEW], alpha=0.16, zorder=3)
    ax.contour(xs, ys, inside.astype(float), levels=[0.5],
               colors=[C_NEW], linewidths=0.5, linestyles=":", zorder=3)


def base(ax, gas, obstacles, title, emitters=()):
    map_w, map_h = MAP_SIZE
    ax.imshow(gas, extent=(0, map_w, 0, map_h), origin="lower", cmap=GAS_CMAP,
              vmin=0, vmax=1.2, interpolation="bilinear", zorder=1)
    ax.imshow(np.ma.masked_where(obstacles < 0.5, obstacles), extent=(0, map_w, 0, map_h),
              origin="lower", cmap=LinearSegmentedColormap.from_list("w", ["#4a4a4a", "#4a4a4a"]),
              interpolation="nearest", zorder=5)
    ax.plot([0, map_w, map_w, 0, 0], [0, 0, map_h, map_h, 0], color="k", lw=0.8, zorder=6)
    for e in emitters:
        ax.plot(*e, marker="s", color="k", ms=2.6, zorder=7)
    ax.set_title(title, pad=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.25, map_w + 0.25)
    ax.set_ylim(-0.25, map_h + 0.25)


def main():
    device = torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    gas, obstacles, geometry = build_scene()

    # --- Inspection: first room, then second room ---
    em_a, em_b = (0.35, 0.35), (0.35, 6.65)
    em_c, em_d = (11.65, 0.35), (11.65, 6.65)
    beams_1 = fan(em_a, 8, 86, 22, geometry) + fan(em_b, -86, -8, 22, geometry)
    beams_2 = fan(em_c, 96, 178, 20, geometry) + fan(em_d, 182, 264, 20, geometry)

    batch_1 = make_batch(beams_1, gas, device)
    batch_2 = make_batch(beams_2, gas, device)

    cfg = Config(env=EnvironmentParams(map_size=MAP_SIZE, cell_size=CELL),
                 sim=SimulationParams(), train=TrainParams(), densify=DensificationParams())
    cfg.device_type = "cpu"
    cfg.quiet = True
    cfg.train.iterations = 700

    environment = EnvironmentContext(
        obstacles=obstacles,
        ground_truth=GroundTruth(gas_map=gas, y_true=batch_1.measurements),
    )

    # --- The model is fitted to the first room alone ---
    model, _ = setup_gs_model(batch_1, cfg)
    trainer = Trainer(model, cfg, environment=environment)
    trainer.train(batch_1)
    model_before = [p.detach().clone() for p in (model._pos, model._scale, model._rotation)]

    # --- The second batch arrives, reaching a plume with no primitives on it ---
    trainer.update_buffers(batch_2)
    beams = trainer.buffer_beams
    with torch.no_grad():
        y_pred = model(beams)
    residuals = trainer.buffer_measurements - y_pred

    min_dist = min(MAP_SIZE) * 0.1
    existing = model.get_pos().detach()
    st = injection_stages(beams, residuals, y_pred, min_dist, existing)

    # The stages above must agree with the function used during training
    reference = extract_candidate_positions(
        beams=beams, importance_scores=residuals, min_dist=min_dist, device=device,
        predictions=y_pred, existing_pos=existing,
    )
    assert torch.allclose(reference, st["points"][st["kept"]].to(device), atol=1e-4), \
        "the drawn stages do not reproduce extract_candidate_positions"

    level = 0.1 * float(gas.max())
    n_first = batch_1.measurements.shape[0]
    beams_np = beams.cpu().numpy()
    score = st["score"].cpu().numpy()
    admitted = st["admitted"].cpu().numpy()
    candidate = st["candidate"].cpu().numpy()
    pts = st["points"].numpy()
    near = st["near"].numpy()
    kept = st["kept"].numpy()
    emitters = (em_a, em_b, em_c, em_d)

    # ------------------------------------------------------------ the figure
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.15), constrained_layout=True)
    ax = axes.ravel()

    # (a) Why the mechanism is needed
    base(ax[0], gas, obstacles, "(a) A new batch reaches an unmodeled region", emitters)
    ax[0].add_collection(LineCollection(beams_np[:n_first], colors=C_OFF, lw=0.35, zorder=2))
    ax[0].add_collection(LineCollection(beams_np[n_first:], colors=C_NEW, lw=0.45, zorder=3))
    model_coverage(ax[0], model, level)
    # (b) The gate: a residual is only actionable where the model holds nothing
    base(ax[1], gas, obstacles, "(b) Candidate beam filtering", emitters)
    ax[1].add_collection(LineCollection(beams_np[~admitted], colors=C_OFF, lw=0.35, zorder=2))
    ax[1].add_collection(LineCollection(beams_np[admitted], colors=C_CAND, lw=0.5, zorder=3))
    model_coverage(ax[1], model, level)

    # (c) The admitted beams ranked by the gas each one misses
    base(ax[2], gas, obstacles, "(c) Path-averaged residual scoring", emitters)
    ax[2].add_collection(LineCollection(beams_np[~admitted], colors=C_OFF, lw=0.35, zorder=2))
    norm = Normalize(0, max(score[admitted].max(), 1e-6))
    ax[2].add_collection(LineCollection(beams_np[admitted],
                                        colors=RES_CMAP(0.25 + 0.75 * norm(score[admitted])),
                                        lw=0.55, zorder=3))

    # (d) Statistical filter among the admitted beams
    base(ax[3], gas, obstacles, "(d) High-residual selection", emitters)
    ax[3].add_collection(LineCollection(beams_np[~candidate], colors=C_OFF, lw=0.35, zorder=2))
    ax[3].add_collection(LineCollection(beams_np[candidate], colors=C_CAND, lw=0.55, zorder=3))

    # (e) Intersections, scored, against the region existing primitives already cover
    base(ax[4], gas, obstacles, "(e) Beam intersection", emitters)
    ax[4].add_collection(LineCollection(beams_np[candidate], colors="#f2b6b6", lw=0.35, zorder=2))
    exclusion_zone(ax[4], existing.cpu().numpy(), min_dist)
    ax[4].scatter(pts[~near, 0], pts[~near, 1], s=6, c=st["point_score"].numpy()[~near],
                  cmap=RES_CMAP, edgecolors="k", linewidths=0.2, zorder=6)
    ax[4].scatter(pts[near, 0], pts[near, 1], s=9, marker="x", color=C_NEW, lw=0.6, zorder=6)
    model_coverage(ax[4], model, level)

    # (f) Non-maximum suppression, and the primitives it produces
    base(ax[5], gas, obstacles, "(f) Injection", emitters)
    for i in kept:
        ax[5].add_patch(Circle(pts[i], 2 * min_dist, fill=False, edgecolor=C_NEW,
                               lw=0.6, ls=(0, (2, 1.5)), zorder=4))
        ax[5].plot(*pts[i], "o", color=C_NEW, ms=1.8, zorder=5)
    model_coverage(ax[5], model, level)

    fig.legend(handles=[
        Line2D([], [], color=C_OFF, lw=1.0, label="beams of previous batches"),
        Line2D([], [], color=C_NEW, lw=1.0, label="arriving batch"),
        Line2D([], [], color=C_MODEL, lw=0.8, ls="--", label="field explained by the model"),
        Line2D([], [], color=C_MODEL, marker=".", ls="", ms=4, label="primitive centers"),
        Line2D([], [], color=C_CAND, lw=1.0, label="selected beams"),
        Line2D([], [], color="k", marker="s", ls="", ms=2.6, label="emitter"),
    ], loc="outside lower center", ncol=6, frameon=False, handlelength=1.3,
        handletextpad=0.5, columnspacing=1.3, fontsize=8.5)

    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/injection_steps.pdf", bbox_inches="tight")
    fig.savefig("plots/injection_steps.png", dpi=300, bbox_inches="tight")

    print(f"primitives after the first room : {model.num_gaussians}")
    print(f"beams in buffer                 : {len(beams_np)} ({n_first} + {len(beams_np) - n_first})")
    print(f"admitted by the gate            : {admitted.sum()} "
          f"({admitted[:n_first].sum()} from the first batch, {admitted[n_first:].sum()} from the second)")
    print(f"candidates after mean+std       : {candidate.sum()}")
    d_near = torch.cdist(st["points"], existing.cpu()).min(dim=1).values
    print(f"intersections                   : {len(pts)} ({near.sum()} discarded by proximity)")
    print(f"  distance to nearest primitive : min {d_near.min():.2f} m, median "
          f"{d_near.median():.2f} m (filter at {min_dist:.2f} m)")
    print(f"injected after NMS              : {len(kept)}")
    print("Saved plots/injection_steps.{pdf,png}")


if __name__ == "__main__":
    main()
