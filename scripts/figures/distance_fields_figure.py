"""Paper figure of the distance fields the obstacle penalty reads.

A U-shaped obstacle (create_obstacle_scenario("u_shape")) and the fields computed
from its occupancy grid by the functions the trainer uses:
  (a) the occupancy grid;
  (b) the ESDF D, with arrows along its gradient in free space (compute_esdf, compute_esdf_gradient);
  (c), (d) the components D_x and D_y of the AA-SDF (compute_aasdf).
Field panels draw only the boundary of the obstacle, so the negative values inside
it stay visible. The color scale is shared by (b)-(d). Where no obstacle lies
along an axis, the AA-SDF component saturates at the map size: it has no real
distance there and does not constrain the primitives, so those cells are drawn
in a neutral grey rather than as a distance. Colors span [-3, 3] m; larger real
distances take the end color.

    python scripts/figures/distance_fields_figure.py [--copy_to ~/paper/figs]
"""
import os
import sys
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.obstacle_utils import create_obstacle_scenario, compute_esdf, compute_esdf_gradient, compute_aasdf

# STIX, as in the other figures of the paper
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

MAP = 10.0          # side of the square map (m)
CELL = 0.1          # cell size (m)
ARROW_STEP = 7      # one gradient arrow every this many cells, in free space
C_WALL = "#4a4a4a"
C_NONE = "0.82"     # AA-SDF cells with no obstacle along the axis
LIMIT = 3.0         # color scale [-LIMIT, LIMIT] m: farther than a few primitive widths, distances no longer matter


def boundary(obstacles: np.ndarray, cell: float) -> LineCollection:
    """Exact cell edges between occupied and free cells, as line segments in meters."""
    occ = obstacles > 0.5
    H, W = occ.shape
    segs = []
    vertical = occ[:, 1:] != occ[:, :-1]          # edge between columns c and c+1
    for r, c in zip(*np.nonzero(vertical)):
        x = (c + 1) * cell
        segs.append([(x, r * cell), (x, (r + 1) * cell)])
    horizontal = occ[1:, :] != occ[:-1, :]        # edge between rows r and r+1
    for r, c in zip(*np.nonzero(horizontal)):
        y = (r + 1) * cell
        segs.append([(c * cell, y), ((c + 1) * cell, y)])
    return LineCollection(segs, colors="k", linewidths=0.9, zorder=4)


def frame(ax, title):
    ax.set_xlim(0, MAP); ax.set_ylim(0, MAP); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, pad=3)


def main():
    parser = argparse.ArgumentParser(description="Figure of the distance fields")
    parser.add_argument("--copy_to", default=None, help="Also copy the PDF here (e.g. the paper's figs folder)")
    args = parser.parse_args()

    n = round(MAP / CELL)
    obstacles = create_obstacle_scenario("u_shape", (n, n))
    esdf = compute_esdf(obstacles, CELL)
    grad = compute_esdf_gradient(esdf)
    aasdf = compute_aasdf(obstacles, CELL, max_dist=MAP)
    extent = (0, MAP, 0, MAP)

    # Saturated AA-SDF cells: no obstacle along that axis
    saturated = np.abs(aasdf) >= MAP - 1e-6
    vmax = float(max(esdf.max(), aasdf[~saturated].max()))
    vmin = float(min(esdf.min(), aasdf[~saturated].min()))
    norm = Normalize(-LIMIT, LIMIT)
    cmap = plt.get_cmap("RdBu")

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05), constrained_layout=True)
    # (a) occupancy grid
    axes[0].imshow(obstacles, origin="lower", extent=extent, cmap=ListedColormap(["white", C_WALL]),
                   interpolation="nearest")
    axes[0].add_collection(boundary(obstacles, CELL))
    frame(axes[0], "(a) Occupancy grid")

    # (b) ESDF and its gradient
    im = axes[1].imshow(esdf, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].add_collection(boundary(obstacles, CELL))
    idx = np.arange(ARROW_STEP // 2, n, ARROW_STEP)
    X, Y = np.meshgrid((idx + 0.5) * CELL, (idx + 0.5) * CELL)
    gx, gy = grad[0][np.ix_(idx, idx)], grad[1][np.ix_(idx, idx)]
    length = np.hypot(gx, gy)
    ok = (length > 1e-6) & (obstacles[np.ix_(idx, idx)] < 0.5)
    axes[1].quiver(X[ok], Y[ok], (gx / np.where(ok, length, 1))[ok], (gy / np.where(ok, length, 1))[ok],
                   color="k", scale=22, width=0.006, headwidth=3.5, headlength=4, zorder=5)
    frame(axes[1], r"(b) ESDF $D$ and $\nabla D$")

    # (c), (d) AA-SDF components
    for ax, k, title in ((axes[2], 0, r"(c) AA-SDF $D_x$"), (axes[3], 1, r"(d) AA-SDF $D_y$")):
        ax.imshow(np.where(saturated[k], np.nan, aasdf[k]), origin="lower", extent=extent, cmap=cmap, norm=norm,
                  interpolation="nearest")
        ax.imshow(np.where(saturated[k], 1.0, np.nan), origin="lower", extent=extent, cmap=ListedColormap([C_NONE]),
                  interpolation="nearest")
        ax.add_collection(boundary(obstacles, CELL))
        frame(ax, title)

    extend = {(False, False): "neither", (True, False): "min", (False, True): "max", (True, True): "both"}[
        (vmin < -LIMIT, vmax > LIMIT)]
    cbar = fig.colorbar(im, ax=axes[1:], fraction=0.03, pad=0.02, extend=extend)
    cbar.set_label("Signed distance (m)")

    os.makedirs("plots", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"plots/distance_fields.{ext}", dpi=300, bbox_inches="tight")
    if args.copy_to:
        os.makedirs(os.path.expanduser(args.copy_to), exist_ok=True)
        shutil.copy("plots/distance_fields.pdf", os.path.expanduser(args.copy_to))
    print(f"color range [{vmin:.2f}, {vmax:.2f}] m; saturated AA-SDF cells: "
          f"{100 * saturated[0].mean():.0f}% of D_x, {100 * saturated[1].mean():.0f}% of D_y")
    print("Saved plots/distance_fields.{pdf,png}")


if __name__ == "__main__":
    main()
