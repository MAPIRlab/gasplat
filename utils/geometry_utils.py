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
