"""Experiment: primitives far from unexplained gas receive no gradient.

Claim under test (Gaussian injection section of the paper): when a new batch of
measurements reaches a region the model holds no primitives in, the gradient those
beams deliver to the existing primitives vanishes, however large their residuals,
so the optimizer alone cannot relocate primitives to the newly observed gas.

Setup. Two plumes of equal strength, one in each half of the domain. Measurements
arrive in two batches: the first only observes the left half, the second sweeps the
right half. Optionally (--wall) a full-height wall separates both halves.

The model is fitted to the first batch once, and that state is duplicated into two
branches that receive the second batch:
  * injection: the full method (splitting, pruning and Gaussian injection);
  * no injection: identical, except that inject_gaussians does nothing.
Both branches start from the same primitives and optimizer state, so injection is
the only difference between them.

Figure:
  (a) the ground truth and both batches;
  (b) the gradient the beams revealing the right plume deliver to a single
      primitive, as it is slid away from that plume. The band marks where the
      primitives fitted to the first batch lie;
  (c, d) the reconstructions of both branches after the second batch.
"""
import copy
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from simple_parsing import ArgumentParser
from shapely.geometry import LineString, Point

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.gaussian_utils import compute_definite_integral
from utils.obstacle_utils import obstacle_mask_to_geometry
from utils.sim_utils import MeasurementBatch, EnvironmentContext, GroundTruth, simulate_gas_integrals

MAP_SIZE = (12.0, 6.0)
CELL = 0.25

# Wall cells [5.75, 6.25] m. Its left face also bounds the first survey without the wall,
# so the first batch is the same in both layouts
WALL_X = (5.75, 6.25)

# (center, sigma, peak) of the plume observed first, and of the one only the second batch reaches
PLUME_LEFT = ((3.0, 3.0), 0.9, 1.0)
PLUME_RIGHT = ((9.0, 3.0), 0.9, 1.0)

BEAMS_PER_FAN = 25


# --------------------------------------------------------------------- scene
def build_scene(wall: bool):
    map_w, map_h = MAP_SIZE
    cols, rows = round(map_w / CELL), round(map_h / CELL)

    xs = (np.arange(cols) + 0.5) * CELL
    ys = (np.arange(rows) + 0.5) * CELL
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)

    gas = np.zeros((rows, cols), dtype=np.float32)
    for (mu, sigma, rho) in (PLUME_LEFT, PLUME_RIGHT):
        gas += rho * np.exp(-0.5 * np.sum((grid - np.asarray(mu)) ** 2, axis=-1) / sigma ** 2)

    obstacles = np.zeros((rows, cols), dtype=np.float32)
    if wall:
        obstacles[:, round(WALL_X[0] / CELL):round(WALL_X[1] / CELL)] = 1.0
    gas[obstacles > 0.5] = 0.0

    geometry = obstacle_mask_to_geometry(obstacles, CELL) if wall else None
    return gas, obstacles, geometry


def fan(origin, a0_deg, a1_deg, geometry, x_range=(0.0, MAP_SIZE[0])):
    """A pan-tilt sweep from one vantage point. Each beam ends at the first wall,
    domain boundary or limit of x_range it meets, where its reflector sits."""
    map_h = MAP_SIZE[1]
    origin = np.asarray(origin, dtype=float)
    beams = []

    for angle in np.deg2rad(np.linspace(a0_deg, a1_deg, BEAMS_PER_FAN)):
        d = np.array([np.cos(angle), np.sin(angle)])

        ts = []
        for comp, bounds in ((0, x_range), (1, (0.0, map_h))):
            if abs(d[comp]) > 1e-9:
                ts += [t for t in ((b - origin[comp]) / d[comp] for b in bounds) if t > 1e-6]
        t_end = min(ts)

        if geometry is not None:
            hit = LineString([tuple(origin), tuple(origin + t_end * d)]).intersection(geometry)
            if not hit.is_empty:
                start = Point(*origin)
                dists = [start.distance(Point(c)) for g in getattr(hit, "geoms", [hit])
                         for c in np.atleast_2d(np.array(g.coords))]
                dists = [dd for dd in dists if dd > 1e-6]
                if dists:
                    t_end = min(t_end, min(dists))

        beams.append((tuple(origin), tuple(origin + t_end * d)))
    return beams


def make_batch(beams, gas, device):
    y = simulate_gas_integrals(gas, beams, CELL, quiet=True)
    return MeasurementBatch(
        beams=torch.tensor(beams, dtype=torch.float32, device=device),
        measurements=torch.tensor(y, dtype=torch.float32, device=device),
    )


def survey_beams(geometry):
    """First batch: four sweeps from the left half, ending at the left face of the
    wall position. Second batch: the mirrored sweeps from the right half, which
    without the wall reach across the whole domain."""
    map_w, map_h = MAP_SIZE
    m = 0.3
    xl, xr = WALL_X[0] - m, WALL_X[1] + m
    left_only = (0.0, WALL_X[0])
    beams_1 = (fan((m, m), 2, 88, geometry, left_only)
               + fan((m, map_h - m), -88, -2, geometry, left_only)
               + fan((xl, m), 92, 178, geometry, left_only)
               + fan((xl, map_h - m), 182, 268, geometry, left_only))
    beams_2 = (fan((map_w - m, m), 92, 178, geometry)
               + fan((map_w - m, map_h - m), 182, 268, geometry)
               + fan((xr, m), 2, 88, geometry)
               + fan((xr, map_h - m), -88, -2, geometry))
    emitters = [(m, m), (m, map_h - m), (xl, m), (xl, map_h - m),
                (map_w - m, m), (map_w - m, map_h - m), (xr, m), (xr, map_h - m)]
    return beams_1, beams_2, emitters


# ------------------------------------------------------------------ training
class SwitchableTrainer(Trainer):
    """Trainer whose Gaussian injection can be switched off."""

    inject = True

    def inject_gaussians(self) -> int:
        return super().inject_gaussians() if self.inject else 0


# --------------------------------------------------------- gradient analysis
# Beams revealing the unmodeled plume are those of the second batch that stay in
# the right half. Beams that also cross the left plume are partly explained by the
# model already, so they are left out.
#
# The gradient of the L1 data term is a sum over beams of w_n = -sign(y_n - y_hat_n)/N
# times the sensitivity of y_hat_n, so the share of a group of beams is that sum
# restricted to the group. A probe primitive faint enough not to flip any residual
# leaves the w_n unchanged, so every probe position is evaluated independently.

def revealing_beams(beams_2):
    return np.array([min(o[0], e[0]) >= WALL_X[0] for o, e in beams_2])


def group_weights(model, trainer, mask):
    """w_n of the beams in mask (aligned with the end of the buffer), 0 elsewhere."""
    with torch.no_grad():
        residual = trainer.buffer_measurements - model(trainer.buffer_beams)
        w = -torch.sign(residual) * trainer.buffer_weights / residual.numel() / model.scale
    keep = torch.zeros_like(w, dtype=torch.bool)
    keep[-len(mask):] = torch.as_tensor(mask, device=w.device)
    return torch.where(keep, w, torch.zeros_like(w))


def parameter_gradient(pos, sigma, theta, rho, w, beams, scale):
    """||dL/dPsi_k|| of each primitive over all its parameters (mu, sigma, theta, rho),
    with rho relative to the concentration scale."""
    params = [t.detach().clone().requires_grad_(True) for t in (pos, sigma, theta, rho)]
    pos, sigma, theta, rho = params
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    cov_inv = R @ torch.diag_embed(1.0 / sigma ** 2) @ R.transpose(1, 2)
    y = compute_definite_integral(pos, cov_inv, rho * scale, beams)
    grads = torch.autograd.grad((w * y).sum(), params)
    flat = torch.cat([g.reshape(g.shape[0], -1) for g in grads], dim=1)
    return flat.norm(dim=1).cpu().numpy()


def gradient_decay(model, trainer, w, dev, step=0.05):
    """Gradient on a probe slid from the right plume to the left edge, along its row.

    The probe has the scale of the plume and a faint concentration. It starts half a
    standard deviation off the plume center, so that it still has a positional
    gradient towards it."""
    (cx, cy), sig, _ = PLUME_RIGHT
    xs = np.arange(cx - 0.5 * sig, 0.0, -step)
    n = len(xs)
    pos = torch.tensor(np.stack([xs, np.full(n, cy)], 1), dtype=torch.float32, device=dev)
    grad = parameter_gradient(pos, torch.full((n, 2), sig, device=dev), torch.zeros(n, device=dev),
                              torch.full((n,), 0.01, device=dev), w, trainer.buffer_beams, model.scale)
    return cx - xs, grad


def existing_gradient(model, trainer, w):
    return parameter_gradient(model.get_pos(), model.get_scale(), model._rotation,
                              model.get_relative_concentration(), w, trainer.buffer_beams, model.scale)


def evaluate(model, gas, obstacles):
    field = model.render_map(CELL)
    free = obstacles < 0.5
    half = gas.shape[1] // 2
    out = {}
    for name, cols in (("left", slice(0, half)), ("right", slice(half, None))):
        f, g, m = field[:, cols], gas[:, cols], free[:, cols]
        out[f"rmse_{name}"] = float(np.sqrt(np.mean((f[m] - g[m]) ** 2)))
        out[f"mass_{name}"] = float(f[m].sum() / max(g[m].sum(), 1e-12))
    out["primitives_right"] = int((model.get_pos()[:, 0] > MAP_SIZE[0] / 2).sum())
    out["primitives"] = model.num_gaussians
    return field, out


# ------------------------------------------------------------------- drawing
def draw_map(ax, img, obstacles, title, vmax):
    map_w, map_h = MAP_SIZE
    ax.imshow(img, extent=(0, map_w, 0, map_h), origin="lower", cmap="viridis",
              vmin=0, vmax=vmax, interpolation="nearest")
    ax.imshow(np.ma.masked_where(obstacles < 0.5, obstacles), extent=(0, map_w, 0, map_h),
              origin="lower", cmap="gray_r", vmin=0, vmax=1.5, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_centers(ax, model):
    with torch.no_grad():
        pos = model.get_pos().cpu().numpy()
    ax.plot(pos[:, 0], pos[:, 1], "+", color="r", ms=4, mew=0.8)


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wall", action="store_true", help="Place a full-height wall between both halves")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="plots")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    layout = "wall" if args.wall else "open"

    cfg = Config(env=EnvironmentParams(map_size=MAP_SIZE, cell_size=CELL),
                 sim=SimulationParams(), train=TrainParams(), densify=DensificationParams())
    cfg.quiet = True
    dev = cfg.device

    gas, obstacles, geometry = build_scene(args.wall)
    beams_1, beams_2, emitters = survey_beams(geometry)
    batch_1 = make_batch(beams_1, gas, dev)
    batch_2 = make_batch(beams_2, gas, dev)
    environment = EnvironmentContext(obstacles=obstacles,
                                     ground_truth=GroundTruth(gas_map=gas, y_true=batch_1.measurements))

    # --- First batch: the left plume only ---
    model, _ = setup_gs_model(batch_1, cfg)
    trainer = SwitchableTrainer(model, cfg, environment=environment)
    trainer.train(batch_1)
    _, metrics_1 = evaluate(model, gas, obstacles)

    # --- Gradient the revealing beams deliver when the second batch arrives ---
    revealing = revealing_beams(beams_2)
    probe = copy.deepcopy(trainer)
    probe.update_buffers(batch_2)
    w = group_weights(probe.model, probe, revealing)
    dist, grad = gradient_decay(probe.model, probe, w, dev)
    grad_rel = grad / grad.max()
    with torch.no_grad():
        prim_dist = (probe.model.get_pos() - torch.tensor(PLUME_RIGHT[0], device=dev)).norm(dim=1).cpu().numpy()
    prim_grad = existing_gradient(probe.model, probe, w)

    # --- Second batch, with and without injection, from the same state ---
    branches = {}
    for name, inject in (("no injection", False), ("injection", True)):
        tr = copy.deepcopy(trainer)
        tr.inject = inject
        tr.train(batch_2)
        branches[name] = (tr, *evaluate(tr.model, gas, obstacles))

    # ------------------------------------------------------------ report
    band = (prim_dist.min(), prim_dist.max())
    in_band = (dist >= band[0]) & (dist <= band[1])
    print(f"Layout: {layout} | beams: {len(beams_1)} + {len(beams_2)} "
          f"({revealing.sum()} revealing the right plume) | device: {dev}")
    print(f"After batch 1: {metrics_1['primitives']} primitives, "
          f"{metrics_1['primitives_right']} in the right half, "
          f"{band[0]:.2f}-{band[1]:.2f} m from the right plume")
    print("Gradient ||dL_data/dPsi|| from the revealing beams:")
    print(f"  probe, relative to its peak, over the band of existing primitives: "
          f"max {grad_rel[in_band].max():.1e}")
    print(f"  probe peak {grad.max():.1e} | existing primitives max {prim_grad.max():.1e}, "
          f"median {np.median(prim_grad):.1e}")
    print(f"{'branch':<14}{'primitives':>11}{'right':>7}{'RMSE L':>9}{'RMSE R':>9}"
          f"{'mass L':>9}{'mass R':>9}")
    for name, (_, _, m) in branches.items():
        print(f"{name:<14}{m['primitives']:>11}{m['primitives_right']:>7}{m['rmse_left']:>9.4f}"
              f"{m['rmse_right']:>9.4f}{m['mass_left']:>9.3f}{m['mass_right']:>9.3f}")
    print("(mass = reconstructed / true gas in each half of the free space)")

    # ------------------------------------------------------------ figure
    vmax = float(gas.max())
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), constrained_layout=True)
    ax = axes.ravel()

    draw_map(ax[0], gas, obstacles, "(a) Ground truth and batches", vmax)
    b2 = np.asarray(beams_2)
    ax[0].add_collection(LineCollection(np.asarray(beams_1), colors="w", lw=0.3, alpha=0.6))
    ax[0].add_collection(LineCollection(b2[revealing], colors="tab:orange", lw=0.3, alpha=0.8))
    if (~revealing).any():
        ax[0].add_collection(LineCollection(b2[~revealing], colors="tab:orange", lw=0.3, alpha=0.25,
                                            linestyles="--"))
    for e in emitters:
        ax[0].plot(*e, "s", color="w" if e[0] < MAP_SIZE[0] / 2 else "tab:orange", ms=3)
    ax[0].legend(handles=[Line2D([], [], color="0.6", lw=1, label="batch 1"),
                          Line2D([], [], color="tab:orange", lw=1, label="batch 2")],
                 loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=7, ncol=2, frameon=False)

    ax[1].axvspan(*band, color="tab:red", alpha=0.15, lw=0, label="existing primitives")
    ax[1].plot(dist, grad_rel, color="k", lw=1.2)
    ax[1].set_yscale("log")
    ax[1].set_ylim(max(grad_rel.min(), 1e-12), 2)
    ax[1].set_xlim(0, dist.max())
    ax[1].set_xlabel("distance to the right plume (m)")
    ax[1].set_ylabel("$\\|\\partial \\mathcal{L}_{data}/\\partial \\Psi\\|$ (relative)")
    ax[1].set_title("(b) Gradient from the batch-2 beams", fontsize=9)
    ax[1].legend(fontsize=7, loc="lower left")
    ax[1].grid(alpha=0.3, which="major", lw=0.3)

    for a, name, tag in ((ax[2], "no injection", "(c)"), (ax[3], "injection", "(d)")):
        tr, field, m = branches[name]
        draw_map(a, field, obstacles,
                 f"{tag} After batch 2, {name} ({100 * m['mass_right']:.0f}% of right plume)", vmax)
        draw_centers(a, tr.model)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"far_field_gradient_{layout}")
    fig.savefig(path + ".png", dpi=200)
    fig.savefig(path + ".pdf")
    print(f"Saved {path}.{{png,pdf}}")


if __name__ == "__main__":
    main()
