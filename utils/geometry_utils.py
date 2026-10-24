import math
import numpy as np
from scipy import sparse

def xy2cell(pos_m: tuple, cell_size_m: float) -> tuple[int, int]:
    """Translate (x, y) coordinates in meters to (row, column) indices."""
    column = int(pos_m[0] // cell_size_m)
    row = int(pos_m[1] // cell_size_m)
    return row, column

def cell2xy(cell_rc: tuple, cell_size_m: float) -> tuple[float, float]:
    """Translate (row, column) cell coordinates to cell-center (x, y) meters."""
    x = cell_rc[1] * cell_size_m + cell_size_m / 2
    y = cell_rc[0] * cell_size_m + cell_size_m / 2
    return x, y

def ray_cell_lengths(
    beams,
    grid_size: tuple[int, int],
    cell_size_m: float,
    chunk: int = 4096,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Length of every beam inside every grid cell, as a sparse (N, rows * cols) matrix.

    Exact traversal: each beam is cut where it crosses the vertical and the
    horizontal grid lines, and every piece is assigned to the cell holding its
    midpoint (cell index row * cols + col, row = y). A piece running along a
    grid line, which only happens for axis-aligned beams, is shared evenly by
    the cells on both sides of it.

    Beams with an endpoint outside the map get an empty row, and are flagged
    False in the returned validity mask (N,).
    """
    beams = np.asarray(beams, dtype=np.float64).reshape(-1, 2, 2)
    rows, cols = grid_size
    width, height = cols * cell_size_m, rows * cell_size_m
    p0, delta = beams[:, 0], beams[:, 1] - beams[:, 0]
    length = np.linalg.norm(delta, axis=1)
    valid = ((beams[..., 0] >= 0) & (beams[..., 0] <= width)
             & (beams[..., 1] >= 0) & (beams[..., 1] <= height)).all(axis=1)

    x_lines = np.arange(cols + 1) * cell_size_m
    y_lines = np.arange(rows + 1) * cell_size_m
    beam_idx, cell_idx, lengths = [], [], []

    for start in range(0, len(beams), chunk):
        sel = np.arange(start, min(start + chunk, len(beams)))
        sel = sel[valid[sel] & (length[sel] > 0)]
        if len(sel) == 0:
            continue
        o, d = p0[sel], delta[sel]

        # Path parameter t in (0, 1) of every crossing with a grid line
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = (x_lines[None, :] - o[:, :1]) / d[:, :1]
            ty = (y_lines[None, :] - o[:, 1:]) / d[:, 1:]
        t = np.concatenate([np.zeros((len(sel), 1)), tx, ty, np.ones((len(sel), 1))], axis=1)
        t[~((t >= 0) & (t <= 1))] = np.inf          # outside the segment, or parallel (nan)
        t.sort(axis=1)

        t_a, t_b = t[:, :-1], t[:, 1:]
        piece = np.isfinite(t_b) & (t_b > t_a)
        b, k = np.nonzero(piece)
        t_mid = (t_a[b, k] + t_b[b, k]) / 2
        seg_len = (t_b[b, k] - t_a[b, k]) * length[sel][b]

        # Cells around the midpoint: the same one twice unless it lies on a grid line
        mid = o[b] + t_mid[:, None] * d[b]
        def neighbours(coord, n):
            f = coord / cell_size_m
            on_line = np.abs(f - np.round(f)) < 1e-9
            lo = np.where(on_line, np.round(f) - 1, np.floor(f))
            hi = np.where(on_line, np.round(f), np.floor(f))
            return np.clip(lo, 0, n - 1).astype(np.int64), np.clip(hi, 0, n - 1).astype(np.int64)
        c_lo, c_hi = neighbours(mid[:, 0], cols)
        r_lo, r_hi = neighbours(mid[:, 1], rows)

        for r in (r_lo, r_hi):
            for c in (c_lo, c_hi):
                beam_idx.append(sel[b])
                cell_idx.append(r * cols + c)
                lengths.append(seg_len / 4)

    if beam_idx:
        beam_idx, cell_idx, lengths = map(np.concatenate, (beam_idx, cell_idx, lengths))
    matrix = sparse.coo_matrix((lengths, (beam_idx, cell_idx)), shape=(len(beams), rows * cols)).tocsr()
    return matrix, valid


def raycast_grid(obstacles: np.ndarray, cell_size: float, origin: tuple[float, float],
                 angle: float, max_range: float) -> float:
    """Distance from `origin` along `angle` to the first blocked cell (DDA).

    obstacles: (H, W) bool, row = y, col = x. Leaving the map counts as a hit.
    Returns the distance to the boundary of the first blocked cell, i.e. the
    beam ends on the wall surface, capped at max_range.
    """
    H, W = obstacles.shape
    x0, y0 = origin
    dx, dy = math.cos(angle), math.sin(angle)
    col, row = int(x0 // cell_size), int(y0 // cell_size)

    step_c = 1 if dx > 0 else -1
    step_r = 1 if dy > 0 else -1
    inv_dx = abs(1.0 / dx) if abs(dx) > 1e-12 else math.inf
    inv_dy = abs(1.0 / dy) if abs(dy) > 1e-12 else math.inf
    next_x = (col + (step_c > 0)) * cell_size
    next_y = (row + (step_r > 0)) * cell_size
    t_max_x = abs(next_x - x0) * inv_dx
    t_max_y = abs(next_y - y0) * inv_dy
    t_delta_x, t_delta_y = cell_size * inv_dx, cell_size * inv_dy

    t = 0.0
    while t < max_range:
        if t_max_x < t_max_y:
            t, col = t_max_x, col + step_c
            t_max_x += t_delta_x
        else:
            t, row = t_max_y, row + step_r
            t_max_y += t_delta_y
        if not (0 <= row < H and 0 <= col < W) or obstacles[row, col]:
            return min(t, max_range)
    return max_range


def visible_cells(origins: np.ndarray, obstacles: np.ndarray, cell_size: float,
                  max_range: float, num_rays: int = 720) -> np.ndarray:
    """Cells crossed by a full 360-degree sweep of beams from each origin.

    Every origin casts `num_rays` evenly spaced rays, marched in quarter-cell
    steps until they hit an obstacle, leave the map or reach max_range. Returns
    an (O, H*W) boolean matrix: entry (o, i) is True when some ray from origin o
    crosses cell i (flat index row * W + col).
    """
    H, W = obstacles.shape
    angles = np.arange(num_rays) * (2 * np.pi / num_rays)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)              # (R, 2)
    O = len(origins)
    pos = np.repeat(origins, num_rays, axis=0)                             # (O*R, 2)
    step = np.tile(dirs, (O, 1)) * (cell_size / 4)
    owner = np.repeat(np.arange(O), num_rays)
    visible = np.zeros((O, H * W), dtype=bool)

    alive = np.arange(O * num_rays)
    for _ in range(int(math.ceil(max_range / (cell_size / 4)))):
        p = pos[alive]
        col = np.floor(p[:, 0] / cell_size).astype(np.int64)
        row = np.floor(p[:, 1] / cell_size).astype(np.int64)
        inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        open_ = inside.copy()
        open_[inside] = ~obstacles[row[inside], col[inside]]
        alive, row, col = alive[open_], row[open_], col[open_]
        if len(alive) == 0:
            break
        visible[owner[alive], row * W + col] = True
        pos[alive] += step[alive]
    return visible
