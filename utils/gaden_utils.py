"""
GADEN -> GASPLAT scenario extraction.

Reads one snapshot (iteration) of a GADEN simulation (filament logfile) and
the 3D occupancy grid straight from a house archive of the VGR dataset
(HouseNN.zip), extracts the horizontal 2D slice of gas concentration at the
beam height, and saves a self-contained "scenario package" that GASPLAT
experiments load without touching GADEN again. The distribution is assumed
quasi-static, so every beam is measured on that single snapshot.

VGR dataset (Ojeda et al., JIRS 2023): 30 houses, each archive holding the
occupancy grid and four gas simulations named after their inlets and outlets
and wind speed, e.g. "2,4-1_slow" (one source each, 2000 iterations saved
every 0.5 s).

Scenario package (one directory per snapshot, named automatically as
<house>_<simulation>_h<height above floor>_it<iteration>, e.g.
house01_2,4-1_slow_h0.85_it1999, under data/scenarios by default):
    scenario.npz   gas_map   (H, W) float32  concentration at the cell centers [ppm]
                   gas_map_fine (H*f, W*f) float32  same slice, f x f samples per cell,
                             used to simulate the measurements (f = meta["fine_factor"])
                   occupancy (H, W) uint8    0 = free, 1 = obstacle, 2 = outlet
    meta.json      cell size, world origin of the local frame, slice height,
                   source position, iteration, ...
    preview.png    occupancy + concentration, for a visual sanity check

Conventions (match the rest of GASPLAT):
    - Maps are (H, W) = (rows, cols) = (y, x). Row 0 is the lowest y.
    - Local metric frame: the map covers [0, W*cs] x [0, H*cs]; cell (r, c)
      has its center at ((c + 0.5) * cs, (r + 0.5) * cs).
    - meta["origin_world"] is the world (GADEN) coordinate of local (0, 0).

Usage:
    python -m utils.gaden_utils HOUSE_ZIP                      # list its simulations
    python -m utils.gaden_utils HOUSE_ZIP SIMULATION \
        [--height 0.85] [--iteration 1999] [--root data/scenarios]
    (height above the floor in meters; the last iteration by default)

Only filament-mode results from GADEN 1.x/2.x (e.g. the VGR dataset) are
supported. Results written by GADEN >= 3 (gaden_core / gaden_gui) start with a
"GADEN_RESULT" header and need gaden_core's own reader.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass

import numpy as np

from utils.paths import SCENARIOS_DIR

FREE, OBSTACLE, OUTLET = 0, 1, 2

# Where scenario packages are written by default (<repo>/data/scenarios)
DEFAULT_ROOT = SCENARIOS_DIR


# ==========================================
#              OCCUPANCY GRID
# ==========================================

@dataclass
class GadenGrid:
    env_min: tuple[float, float, float]
    env_max: tuple[float, float, float]
    shape: tuple[int, int, int]          # (nx, ny, nz)
    cell_size: float
    cells: np.ndarray                    # (nx, ny, nz) uint8: 0 free, 1 obstacle, 2 outlet

    @property
    def free(self) -> np.ndarray:
        return self.cells == FREE

    def layer_index(self, height: float) -> int:
        """Layer containing world height `height` (same rule as GADEN)."""
        k = int((height - self.env_min[2]) / self.cell_size)
        if not 0 <= k < self.shape[2]:
            raise ValueError(f"Height {height} m is outside the environment "
                             f"[{self.env_min[2]}, {self.env_max[2]}] m.")
        return k

    def floor_height(self) -> float:
        """World height of the bottom of the lowest layer with free cells."""
        k = int(np.argmax(self.free.reshape(-1, self.shape[2]).any(axis=0)))
        return self.env_min[2] + k * self.cell_size


def parse_occupancy(text: str) -> GadenGrid:
    """Parse the contents of GADEN's OccupancyGrid3D.csv.

    Header: 4 comment lines (#env_min, #env_max, #num_cells, #cell_size).
    Body: one block per z layer, separated by ';' lines. Inside a block, each
    line is one x index and each value one y index.
    """
    lines = text.splitlines()

    def header_values(prefix):
        line = next(l for l in lines[:4] if l.startswith(prefix))
        return [float(v) for v in line.split()[1:]]

    env_min = tuple(header_values("#env_min"))
    env_max = tuple(header_values("#env_max"))
    nx, ny, nz = (int(v) for v in header_values("#num_cells"))
    cell_size = header_values("#cell_size")[0]

    cells = np.empty((nx, ny, nz), dtype=np.uint8)
    k, i = 0, 0
    for line in lines[4:]:
        line = line.strip()
        if not line:
            continue
        if line == ";":
            k, i = k + 1, 0
            continue
        cells[i, :, k] = np.array(line.split(), dtype=np.uint8)
        i += 1
    if k < nz - 1:
        raise ValueError(f"Occupancy file has {k + 1} layers, header says {nz}.")

    return GadenGrid(env_min, env_max, (nx, ny, nz), cell_size, cells)


# ==========================================
#              FILAMENT LOGFILES
# ==========================================

@dataclass
class GadenIteration:
    filaments: np.ndarray            # (N, 4): x, y, z [m], sigma [cm]
    source_position: tuple[float, float, float]
    total_moles_in_filament: float
    num_moles_all_gases_in_cm3: float
    wind_iteration: int


def parse_iteration(raw: bytes) -> GadenIteration:
    """Parse one GADEN filament logfile (format versions 1 and 2.x).

    The environment bounds stored in some version-1 headers are corrupted
    (this affects the VGR dataset), so they are skipped here and taken from
    the occupancy file instead.
    """
    if raw.startswith(b"GADEN_RESULT"):
        raise NotImplementedError(
            "Logfile written by GADEN >= 3 (gaden_core/gaden_gui). "
            "Read it with gaden_core's PlaybackSimulation instead.")
    data = zlib.decompress(raw)

    major = struct.unpack_from("=i", data, 0)[0]
    if major == 1:
        off = 4 + 48 + 12                 # version, env_min/max (6 doubles), num_cells
        off += 3 * 8                      # cell size (+2 repeated copies)
        src = struct.unpack_from("=ddd", data, off); off += 24
        off += 4                          # gas type
        total_moles, moles_air = struct.unpack_from("=dd", data, off); off += 16
        wind_it = struct.unpack_from("=i", data, off)[0]; off += 4
        rec, fil_fmt = struct.calcsize("=idddd"), "=dddd"
    elif major == 2:
        minor = struct.unpack_from("=i", data, 4)[0]
        off = 8 + 12 + 24 + 4             # versions, num_cells, env_min/max (6 floats), cell size
        src = struct.unpack_from("=fff", data, off); off += 12
        off += 4                          # gas type
        if minor < 6:
            total_moles, moles_air = struct.unpack_from("=dd", data, off); off += 16
            fil_fmt = "=dddd"
        else:
            total_moles, moles_air = struct.unpack_from("=ff", data, off); off += 8
            fil_fmt = "=ffff"
        wind_it = struct.unpack_from("=i", data, off)[0]; off += 4
        rec = 4 + struct.calcsize(fil_fmt)
    else:
        raise ValueError(f"Unknown GADEN logfile version {major}")

    n = (len(data) - off) // rec
    dtype = np.float64 if fil_fmt == "=dddd" else np.float32
    body = np.frombuffer(data, dtype=np.uint8, count=n * rec, offset=off).reshape(n, rec)
    filaments = body[:, 4:].copy().view(dtype).reshape(n, 4).astype(np.float64)

    return GadenIteration(filaments, tuple(src), float(total_moles), float(moles_air), int(wind_it))


# ==========================================
#              VGR HOUSE ARCHIVES
# ==========================================

class VGRHouse:
    """One house of the VGR dataset, read straight from its zip archive.

    The archive holds HouseNN/OccupancyGrid3D.csv and, for every simulation,
    HouseNN/gas_simulations/<simulation>/<results dir>/iteration_<n>.
    """

    def __init__(self, zip_path: str):
        self.path = zip_path
        self.zip = zipfile.ZipFile(zip_path)
        self.name = os.path.splitext(os.path.basename(zip_path))[0]
        self._results: dict[str, str] = {}           # simulation -> results dir
        self._iterations: dict[str, list[int]] = {}
        pattern = re.compile(rf"{re.escape(self.name)}/gas_simulations/([^/]+)/([^/]+)/iteration_(\d+)")
        for member in self.zip.namelist():
            if m := pattern.fullmatch(member):
                sim = m.group(1)
                self._results[sim] = f"{self.name}/gas_simulations/{sim}/{m.group(2)}"
                self._iterations.setdefault(sim, []).append(int(m.group(3)))
        for its in self._iterations.values():
            its.sort()

    @property
    def simulations(self) -> list[str]:
        return sorted(self._results)

    def iterations(self, simulation: str) -> list[int]:
        return self._iterations[self._check(simulation)]

    def occupancy(self) -> GadenGrid:
        return parse_occupancy(self.zip.read(f"{self.name}/OccupancyGrid3D.csv").decode())

    def iteration(self, simulation: str, n: int) -> GadenIteration:
        return parse_iteration(self.zip.read(f"{self._results[self._check(simulation)]}/iteration_{n}"))

    def _check(self, simulation: str) -> str:
        if simulation not in self._results:
            raise KeyError(f"{self.name} has no simulation '{simulation}'. "
                           f"Available: {', '.join(self.simulations)}")
        return simulation


# ==========================================
#           CONCENTRATION SLICES
# ==========================================

def concentration_slice(grid: GadenGrid, it: GadenIteration, k: int,
                        cutoff_sigmas: float = 5.0, fine_factor: int = 1):
    """Concentration [ppm] at the centers of the cells of layer k.

    Re-implements GADEN's player model, vectorized per filament:
      - each filament is an isotropic 3D Gaussian (sigma in cm),
      - it contributes only within `cutoff_sigmas` sigmas (GADEN uses 5),
      - and only if the straight path to the cell crosses no obstacle,
        checked by stepping one cell size at a time, as GADEN does.
    Obstacle and outlet cells are set to 0.

    With fine_factor f > 1, the slice is also sampled at f x f points per cell
    (sub-cell centers). Their line of sight is the one of their cell, while their
    distance to each filament is exact.

    Returns an (nx, ny) array (GADEN's x/y indexing, not GASPLAT's (H, W)), or a
    tuple (centers (nx, ny), fine (nx*f, ny*f)) when fine_factor > 1.
    """
    nx, ny, nz = grid.shape
    cs = grid.cell_size
    emin = np.asarray(grid.env_min)
    free = grid.free
    out = np.zeros((nx, ny))
    z_c = emin[2] + (k + 0.5) * cs
    f = fine_factor
    out_fine = np.zeros((nx * f, ny * f)) if f > 1 else None
    sub = (np.arange(f) + 0.5) / f - 0.5            # sub-cell offsets, in cells

    # GADEN's gaussian normalization, folded into one constant (ppm units)
    scale = it.total_moles_in_filament / it.num_moles_all_gases_in_cm3 * 1e6 / math.sqrt(8 * math.pi ** 3)

    for x_f, y_f, z_f, sigma_cm in it.filaments:
        r = cutoff_sigmas * sigma_cm / 100.0           # cutoff radius [m]
        if abs(z_f - z_c) > r:
            continue
        # window of cells whose centers may be within r
        i0 = max(int(math.floor((x_f - r - emin[0]) / cs)), 0)
        i1 = min(int(math.ceil((x_f + r - emin[0]) / cs)), nx)
        j0 = max(int(math.floor((y_f - r - emin[1]) / cs)), 0)
        j1 = min(int(math.ceil((y_f + r - emin[1]) / cs)), ny)
        if i0 >= i1 or j0 >= j1:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        loc = np.stack([emin[0] + (ii + 0.5) * cs,
                        emin[1] + (jj + 0.5) * cs,
                        np.full(ii.shape, z_c)], axis=1)
        delta = np.array([x_f, y_f, z_f]) - loc
        dist = np.linalg.norm(delta, axis=1)
        keep = (dist <= r) & free[ii, jj, k]
        if not keep.any():
            continue
        ii, jj, loc, delta, dist = ii[keep], jj[keep], loc[keep], delta[keep], dist[keep]

        # line of sight, stepping like GADEN: steps = int(d / cs), check each step
        steps = (dist / cs).astype(int)
        visible = np.ones(len(dist), dtype=bool)
        max_steps = steps.max()
        if max_steps > 0:
            t = np.arange(1, max_steps + 1)
            frac = t[None, :] / np.maximum(steps, 1)[:, None]                  # (M, S)
            pts = loc[:, None, :] + frac[..., None] * delta[:, None, :]          # (M, S, 3)
            idx = ((pts - emin) / cs).astype(int)                                # truncation, as in C
            for a, n_a in enumerate((nx, ny, nz)):
                np.clip(idx[..., a], 0, n_a - 1, out=idx[..., a])
            blocked = ~free[idx[..., 0], idx[..., 1], idx[..., 2]]
            blocked &= t[None, :] <= steps[:, None]
            visible = ~blocked.any(axis=1)

        d_cm = 100.0 * dist[visible]
        c = scale / sigma_cm ** 3 * np.exp(-d_cm ** 2 / (2 * sigma_cm ** 2))
        np.add.at(out, (ii[visible], jj[visible]), c)

        if f > 1:
            # Sub-cell points of the visible cells: (M, f, f) positions and indices
            vi, vj, vloc = ii[visible], jj[visible], loc[visible]
            px = vloc[:, 0, None, None] + sub[None, :, None] * cs
            py = vloc[:, 1, None, None] + sub[None, None, :] * cs
            d_fine = 100.0 * np.sqrt((x_f - px) ** 2 + (y_f - py) ** 2 + (z_f - z_c) ** 2)
            c_fine = np.where(d_fine <= 100.0 * r, scale / sigma_cm ** 3 * np.exp(-d_fine ** 2 / (2 * sigma_cm ** 2)), 0.0)
            fi = vi[:, None, None] * f + np.arange(f)[None, :, None]
            fj = vj[:, None, None] * f + np.arange(f)[None, None, :]
            np.add.at(out_fine, (np.broadcast_to(fi, c_fine.shape).ravel(), np.broadcast_to(fj, c_fine.shape).ravel()),
                      c_fine.ravel())

    return (out, out_fine) if f > 1 else out


# ==========================================
#            SCENARIO EXTRACTION
# ==========================================

def _to_map(a_xy: np.ndarray) -> np.ndarray:
    """GADEN (x, y) indexing -> GASPLAT (H, W) = (y, x) indexing."""
    return np.ascontiguousarray(a_xy.T)


def scenario_name(house: str, simulation: str, height_above_floor: float, iteration: int) -> str:
    """Directory name of a scenario package, e.g. house01_2,4-1_slow_h0.85_it1999."""
    return f"{house.lower()}_{simulation}_h{height_above_floor:.2f}_it{iteration}"


def extract_scenario(house_zip: str, simulation: str, root: str = DEFAULT_ROOT,
                     height: float = 0.85, iteration: int | None = None,
                     crop_margin: int = 1, cutoff_sigmas: float = 5.0,
                     fine_factor: int = 4) -> tuple[str, dict]:
    """Build a scenario package from one snapshot of a VGR simulation.

    The package is written to root/<scenario_name(...)>, named after the
    height of the slice actually taken, and its path is returned with the
    metadata.

    height: height of the beam plane above the floor [m]. The slice is the
    occupancy layer containing it (0.1 m layers in VGR).
    iteration: which snapshot to slice; the last one saved by default.
    crop_margin: cells kept around the free space when cropping the map.
    """
    house = VGRHouse(house_zip)
    if iteration is None:
        iteration = house.iterations(simulation)[-1]
    grid = house.occupancy()
    k = grid.layer_index(grid.floor_height() + height)

    # --- 2D occupancy of the beam layer, cropped to the free space ---
    layer = grid.cells[:, :, k]
    free_xy = np.argwhere(layer == FREE)
    if len(free_xy) == 0:
        raise ValueError(f"Layer {k} ({height} m above the floor) has no free cells.")
    lo = np.maximum(free_xy.min(axis=0) - crop_margin, 0)
    hi = np.minimum(free_xy.max(axis=0) + crop_margin + 1, layer.shape)
    crop = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
    occupancy = _to_map(layer[crop])

    # --- concentration slice ---
    it = house.iteration(simulation, iteration)
    slices = concentration_slice(grid, it, k, cutoff_sigmas, fine_factor)
    centers, fine = slices if fine_factor > 1 else (slices, None)
    gas_map = _to_map(centers[crop]).astype(np.float32)
    # Finer samples of the same slice, to simulate the measurements on
    arrays = {"gas_map": gas_map, "occupancy": occupancy}
    if fine is not None:
        f = fine_factor
        arrays["gas_map_fine"] = _to_map(fine[lo[0] * f:hi[0] * f, lo[1] * f:hi[1] * f]).astype(np.float32)
    source = it.source_position

    cs = grid.cell_size
    origin_world = (grid.env_min[0] + lo[0] * cs, grid.env_min[1] + lo[1] * cs)
    meta = {
        "house_zip": os.path.abspath(house_zip),
        "house": house.name,
        "simulation": simulation,
        "iteration": iteration,
        "num_filaments": len(it.filaments),
        "cell_size": cs,
        "map_size_m": [occupancy.shape[1] * cs, occupancy.shape[0] * cs],   # (width, height)
        "grid_shape_hw": list(occupancy.shape),
        "origin_world": list(origin_world),
        "layer_index": k,
        "layer_center_z": grid.env_min[2] + (k + 0.5) * cs,
        "height_above_floor": grid.env_min[2] + (k + 0.5) * cs - grid.floor_height(),
        "source_world": list(source),
        "source_local": [source[0] - origin_world[0], source[1] - origin_world[1]],
        "units": "ppm (maps), ppm*m (beam integrals)",
        "cutoff_sigmas": cutoff_sigmas,
        "fine_factor": fine_factor,
        "occupancy_codes": {"free": FREE, "obstacle": OBSTACLE, "outlet": OUTLET},
    }

    meta["name"] = scenario_name(house.name, simulation, meta["height_above_floor"], iteration)
    out_dir = os.path.join(root, meta["name"])
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "scenario.npz"), **arrays)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    save_preview(os.path.join(out_dir, "preview.png"), gas_map, occupancy, meta)
    return out_dir, meta


@dataclass
class Scenario:
    gas_map: np.ndarray       # (H, W) [ppm], at the cell centers: ground truth for evaluation
    occupancy: np.ndarray     # (H, W) uint8 codes
    meta: dict
    gas_map_fine: np.ndarray | None = None   # (H*f, W*f) [ppm], f = meta["fine_factor"]: simulates the measurements

    @property
    def obstacles(self) -> np.ndarray:
        """Boolean (H, W) mask for GASPLAT: obstacles and outlets both block."""
        return self.occupancy != FREE


def load_scenario(path: str) -> Scenario:
    data = np.load(os.path.join(path, "scenario.npz"))
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    fine = data["gas_map_fine"] if "gas_map_fine" in data.files else None
    return Scenario(data["gas_map"], data["occupancy"], meta, fine)


def save_preview(path, gas_map, occupancy, meta, beams=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = meta["cell_size"]
    H, W = occupancy.shape
    extent = (0, W * cs, 0, H * cs)
    fig, ax = plt.subplots(figsize=(7, 7 * H / W + 0.6))
    shown = np.ma.masked_where(occupancy != FREE, gas_map)
    im = ax.imshow(shown, origin="lower", extent=extent, cmap="viridis")
    ax.imshow(np.ma.masked_where(occupancy == FREE, occupancy), origin="lower",
              extent=extent, cmap="Greys", vmin=0, vmax=2.5, interpolation="nearest")
    if beams is not None:
        for (x0, y0), (x1, y1) in beams:
            ax.plot([x0, x1], [y0, y1], color="tab:orange", lw=0.5, alpha=0.6)
    sx, sy = meta["source_local"]
    ax.plot(sx, sy, marker="*", color="red", ms=12, mec="white")
    snapshot = f"Iteration {meta['iteration']}" if "iteration" in meta else f"t = {meta['time_s']:.0f} s"
    ax.set_title(f"{snapshot}, z = {meta['layer_center_z']:.2f} m "
                 f"({meta['height_above_floor']:.2f} m above floor)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, label="ppm", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract a GASPLAT scenario from a VGR house archive.")
    p.add_argument("house_zip", help="HouseNN.zip of the VGR dataset")
    p.add_argument("simulation", nargs="?", help="e.g. 2,4-1_slow (omit to list them)")
    p.add_argument("--height", type=float, default=0.85, help="beam plane above the floor [m]")
    p.add_argument("--iteration", type=int, default=None, help="snapshot to extract (default: the last one)")
    p.add_argument("--root", default=DEFAULT_ROOT, help="directory for the scenario package (default: data/scenarios)")
    args = p.parse_args()

    if args.simulation is None:
        house = VGRHouse(args.house_zip)
        for sim in house.simulations:
            its = house.iterations(sim)
            src = house.iteration(sim, its[0]).source_position
            print(f"{sim:16s} {len(its)} iterations ({its[0]}-{its[-1]}), "
                  f"source ({src[0]:.2f}, {src[1]:.2f}, {src[2]:.2f})")
    else:
        out_dir, meta = extract_scenario(args.house_zip, args.simulation, args.root,
                                         args.height, args.iteration)
        print(f"Scenario written to {out_dir}")
        if abs(meta["height_above_floor"] - args.height) > 1e-6:
            print(f"Note: the slice is the layer centered {meta['height_above_floor']:.2f} m above the floor "
                  f"(requested {args.height} m)")
        print(json.dumps({k: meta[k] for k in ("grid_shape_hw", "map_size_m", "layer_center_z",
                                               "height_above_floor", "source_local", "num_filaments")}, indent=2))
