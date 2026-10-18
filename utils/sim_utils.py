import math
import torch
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Point

from config import Config
from utils.geometry_utils import ray_cell_lengths


@dataclass
class MeasurementBatch:
    """Sensor readings and their geometries."""
    beams: torch.Tensor         # Shape: (N, 2, 2)
    measurements: torch.Tensor  # Shape: (N,)
    pose_ids: torch.Tensor | None = None  # Shape: (N,) - vantage pose of each beam, if any

@dataclass
class GroundTruth:
    """Holds perfect knowledge of the world."""
    gas_map: np.ndarray         # Shape: (H, W)
    y_true: torch.Tensor        # Shape: (N,) - Noise-free measurements

@dataclass
class EnvironmentContext:
    """Static physical properties of the environment."""
    obstacles: np.ndarray | None = None
    ground_truth: GroundTruth | None = None


# ==========================================
#           GAS DISTRIBUTION
# ==========================================

def generate_fractal_gas_distribution(
        grid_size, scale_fraction=0.2, octaves=3, 
        threshold=0.3, power=2.0, center_bias=2.0,
        rng: np.random.Generator | None = None):
    """
    Generates center-biased, scale-invariant gas distribution using fractal noise.
    
    - scale_fraction: Size of base blobs relative to map size (e.g., 0.15 = 15%).
    - center_bias: How aggressively to pull clouds to the center. 
                   0.0 = no bias (everywhere), >2.0 = tightly packed in center.
    - rng: Random generator. A fresh, unseeded one if None.
    """
    rng = rng or np.random.default_rng()
    height, width = grid_size
    
    # Dynamic Scale
    base_scale = min(height, width) * scale_fraction
    
    # Generate Fractal Noise
    noise = np.zeros(grid_size)
    frequency, amplitude = 1.0, 1.0
    
    for _ in range(octaves):
        base = rng.random(grid_size)
        smoothed = gaussian_filter(base, sigma=base_scale / frequency)
        noise += smoothed * amplitude
        amplitude *= 0.5
        frequency *= 2.0
        
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    # Create Gaussian Envelope (Center Mask)
    if center_bias > 0.0:
        y, x = np.indices(grid_size)
        center_y, center_x = height / 2.0, width / 2.0
        
        # Calculate squared distance from the center for every cell
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        max_dist_sq = (width / 2.0)**2 + (height / 2.0)**2
        
        # Create the mask: e^(-(distance^2) / variance)
        variance = max_dist_sq / center_bias
        mask = np.exp(-dist_sq / variance)
        
        # Apply the envelope to force edges towards zero
        noise = noise * mask

    # Threshold and Power Curve
    gas_map = np.maximum(0, noise - threshold)
    
    if gas_map.max() > 0:
        gas_map = gas_map / gas_map.max()
        
    gas_map = gas_map ** power
    
    return gas_map

# ==========================================
#            BEAM FUNCTIONS
# ==========================================

def generate_radial_beams(map_size_m: tuple, num_beams: int):
    """Generates beams starting from bottom corners with endpoints distributed homogeneously in angle."""
    beams = []
    map_x, map_y = map_size_m

    num_beams_left = int(num_beams // 2)
    if num_beams_left > 0:
        angles_left = np.linspace(0, np.pi/2, num_beams_left)
        for angle in angles_left:
            x0, y0 = 0.0, 0.0
            if angle == 0:
                x1, y1 = map_x, 0.0
            elif angle == np.pi / 2:
                x1, y1 = 0.0, map_y
            else:
                if angle <= math.atan2(map_y, map_x):
                    x1 = map_x
                    y1 = map_x * np.tan(angle)
                else:
                    x1 = map_y / np.tan(angle)
                    y1 = map_y
            beams.append(((x0, y0), (x1, y1)))

    num_beams_right = num_beams - num_beams_left
    if num_beams_right > 0:
        angles_right = np.linspace(0, np.pi/2, num_beams_right)
        for angle in angles_right:
            x0, y0 = map_x, 0.0
            if angle == 0:
                x1, y1 = 0.0, 0.0
            elif angle == np.pi / 2:
                x1, y1 = map_x, map_y
            else:
                if angle <= math.atan2(map_y, map_x):
                    x1 = 0.0
                    y1 = map_x * np.tan(angle)
                else:
                    x1 = map_x - (map_y / np.tan(angle))
                    y1 = map_y
            beams.append(((x0, y0), (x1, y1)))
    
    return beams

def generate_random_beams(map_size_m: tuple, num_beams: int, rng: np.random.Generator | None = None):
    """Generates random TDLAS beams from the perimeter of a map in meters."""
    rng = rng or np.random.default_rng()
    beams = []
    map_x, map_y = map_size_m

    for _ in range(num_beams):
        start_edge = rng.choice(['left', 'right', 'bottom', 'top'])
        if start_edge == 'left':
            x0, y0 = 0.0, rng.uniform(0, map_y)
        elif start_edge == 'right':
            x0, y0 = map_x, rng.uniform(0, map_y)
        elif start_edge == 'bottom':
            x0, y0 = rng.uniform(0, map_x), 0.0
        else:
            x0, y0 = rng.uniform(0, map_x), map_y

        end_edges = [edge for edge in ['left', 'right', 'bottom', 'top'] if edge != start_edge]
        end_edge = rng.choice(end_edges)
        if end_edge == 'left':
            x1, y1 = 0.0, rng.uniform(0, map_y)
        elif end_edge == 'right':
            x1, y1 = map_x, rng.uniform(0, map_y)
        elif end_edge == 'bottom':
            x1, y1 = rng.uniform(0, map_x), 0.0
        else:
            x1, y1 = rng.uniform(0, map_x), map_y

        beams.append(((x0, y0), (x1, y1)))
        
    return beams

def generate_horizontal_vertical_beams(map_size_m: tuple, num_beams: int):
    """Generates half horizontal and half vertical beams, evenly distributed."""
    beams = []
    map_x, map_y = map_size_m
    
    h_beams = int(num_beams*map_y // (map_x+map_y))

    if h_beams > 0:
        y_positions = np.linspace(0, map_y, h_beams, endpoint=False)
        for y in y_positions:
            beams.append(((0.0, y), (map_x, y)))
            
    remaining_beams = num_beams - h_beams
    if remaining_beams > 0:
        x_positions = np.linspace(0, map_x, remaining_beams, endpoint=False)
        for x in x_positions:
            beams.append(((x, 0.0), (x, map_y)))

    return beams


def generate_simple_beams(map_size_m: tuple[float, float], num_beams: int,
                          rng: np.random.Generator | None = None):
    """Generate a lightweight mixed beam set using the simple beam layouts already in this module."""
    if num_beams <= 0:
        return []

    generators = [
        generate_random_beams,
        generate_radial_beams,
        generate_horizontal_vertical_beams,
    ]

    counts = [num_beams // len(generators)] * len(generators)
    for idx in range(num_beams % len(generators)):
        counts[idx] += 1

    beams = []
    for generator, count in zip(generators, counts):
        if count > 0:
            if generator is generate_random_beams:
                beams.extend(generator(map_size_m, count, rng=rng))
            else:
                beams.extend(generator(map_size_m, count))

    return beams[:num_beams]


def _intersection_points(geometry):
    """Yield representative points from a ray/obstacle intersection."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "Point":
        yield geometry
    elif geometry.geom_type in {"MultiPoint", "GeometryCollection"}:
        for item in geometry.geoms:
            yield from _intersection_points(item)
    elif geometry.geom_type in {"LineString", "MultiLineString"}:
        for item in geometry.geoms if hasattr(geometry, "geoms") else [geometry]:
            yield Point(item.coords[0])
            yield Point(item.coords[-1])

def generate_obstacle_aware_beams(
    map_size_m: tuple[float, float],
    num_beams: int,
    obstacle_geometry=None,
    rng: np.random.Generator | None = None,
) -> list:
    """Generate beam geometries from the existing simple layout and clip them at obstacles."""
    if num_beams <= 0:
        return []
    rng = rng or np.random.default_rng()

    if obstacle_geometry is None:
        return generate_simple_beams(map_size_m, num_beams, rng=rng)

    beams = []
    candidate_count = max(num_beams * 10, 100)
    candidate_beams = generate_simple_beams(map_size_m, candidate_count, rng=rng)
    rng.shuffle(candidate_beams)

    for (x0, y0), (x1, y1) in candidate_beams:
        line = LineString([(x0, y0), (x1, y1)])
        total_length = line.length
        if total_length <= 1e-8:
            continue

        start_point = Point(x0, y0)
        if obstacle_geometry.covers(start_point):
            continue

        closest_distance = total_length
        for point in _intersection_points(line.intersection(obstacle_geometry)):
            distance = start_point.distance(point)
            if 1e-6 < distance < closest_distance:
                closest_distance = distance

        if closest_distance < total_length:
            clipped = line.interpolate(closest_distance)
            endpoint = (clipped.x, clipped.y)
        else:
            endpoint = (x1, y1)

        beams.append(((x0, y0), endpoint))
        if len(beams) >= num_beams:
            return beams[:num_beams]

    if len(beams) < num_beams:
        raise RuntimeError(
            f"Could only generate {len(beams)} obstacle-aware beams out of {num_beams}."
        )
    return beams[:num_beams]


# ==========================================
#          VANTAGE-POINT BEAMS
# ==========================================

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


def select_vantage_poses(obstacles: np.ndarray, cell_size: float,
                         num_poses: int | None = None, min_clearance: float = 0.3,
                         max_range: float = 30.0, min_angle: float = 30.0,
                         coverage: float = 0.95, spacing: float = 0.3) -> np.ndarray:
    """Place robot poses so that the free space is seen from two directions.

    A beam integral only localizes gas along its own line, so a region is
    resolved where beams from at least two poses cross it. A free cell counts
    as covered once it is crossed by beams from two poses whose lines differ by
    at least `min_angle` degrees (lines, not rays: opposite poses see a cell
    along the same line and do not count).

    Poses are chosen greedily among candidate positions, free cells on a grid
    of `spacing` m with at least `min_clearance` m to any obstacle, in the
    largest connected free region, so the poses depend on the map alone.
    Each step adds the candidate that increases
    sum_i level_i the most, where level_i is 0 for an unseen cell, 1 for a cell
    seen from a single direction and 2 for a covered one. Selection stops after
    `num_poses` poses or, if num_poses is None, once a fraction `coverage` of the
    free cells is covered (or no candidate improves it further).

    Returns (P, 2) positions in meters.
    """
    from scipy.ndimage import distance_transform_edt, label

    H, W = obstacles.shape
    free = ~obstacles.astype(bool)
    labels, n = label(free)
    if n == 0:
        raise ValueError("The map has no free space.")
    reachable = labels == 1 + np.argmax(np.bincount(labels.ravel())[1:])
    # pad with obstacles so the map border also counts for clearance
    # (center-to-center EDT minus half a cell = distance to the obstacle surface)
    clearance = (distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] - 0.5) * cell_size

    stride = max(1, round(spacing / cell_size))
    grid = np.zeros_like(free)
    grid[stride // 2::stride, stride // 2::stride] = True
    rows, cols = np.nonzero(grid & reachable & (clearance >= min_clearance))
    if len(rows) == 0:
        raise ValueError(f"No candidate pose with {min_clearance} m clearance.")
    cand = np.stack([(cols + 0.5) * cell_size, (rows + 0.5) * cell_size], axis=1)

    target = reachable.ravel()
    vis = visible_cells(cand, obstacles, cell_size, max_range) & target   # (C, H*W)
    cy, cx = np.divmod(np.arange(H * W), W)
    cell_xy = np.stack([(cx + 0.5) * cell_size, (cy + 0.5) * cell_size], axis=1)
    # direction of the line from each candidate to each cell, in [0, pi)
    bearing = np.mod(np.arctan2(cell_xy[None, :, 1] - cand[:, None, 1],
                                cell_xy[None, :, 0] - cand[:, None, 0]), np.pi)

    min_angle = math.radians(min_angle)
    level = np.zeros(H * W, dtype=np.int8)
    # crosses[c, i]: candidate c would see cell i at min_angle or more from a chosen pose
    crosses = np.zeros_like(vis)
    chosen = []
    num_target = int(target.sum())
    while num_poses is None or len(chosen) < num_poses:
        gain = (vis & (level == 0)).sum(axis=1) + (vis & crosses & (level == 1)).sum(axis=1)
        best = int(np.argmax(gain))
        if gain[best] == 0:
            if num_poses is not None:
                raise ValueError(f"Only {len(chosen)} poses improve the coverage, "
                                 f"{num_poses} requested.")
            break
        chosen.append(best)
        seen = vis[best]
        level[seen & (level == 0)] = 1
        level[seen & crosses[best]] = 2
        diff = np.abs(bearing - bearing[best])
        crosses |= seen & (np.minimum(diff, np.pi - diff) >= min_angle)
        if num_poses is None and (level == 2).sum() >= coverage * num_target:
            break
    return cand[chosen]


def sweep_angles(heading: float, angular_step: float, fov: float = 360.0) -> np.ndarray:
    """Beam directions (rad) of one pan sweep with a fixed angular resolution.

    angular_step and fov are in degrees. A full turn takes round(360 / step)
    beams spread evenly around it, so a step that does not divide 360 is
    adjusted slightly to close the circle. A partial sweep keeps the step
    exactly and is centered on `heading` (rad).
    """
    if fov >= 360.0 - 1e-9:
        n = max(1, round(360.0 / angular_step))
        return heading + np.arange(n) * (2 * np.pi / n)
    n = int(fov / angular_step + 1e-9) + 1
    return heading + np.radians((np.arange(n) - (n - 1) / 2) * angular_step)


def generate_vantage_beams(obstacles: np.ndarray, cell_size: float, angular_step: float,
                           num_poses: int | None = None, poses: np.ndarray | None = None,
                           min_clearance: float = 0.3, max_range: float = 30.0,
                           min_angle: float = 30.0, coverage: float = 0.95,
                           min_length: float = 0.2, fov: float = 360.0,
                           rng: np.random.Generator | None = None):
    """Beams shot by a robot that stops at several poses and pans its TDLAS.

    Poses come from select_vantage_poses unless given. At each pose the sensor
    sweeps `fov` degrees with one beam every `angular_step` degrees, as a
    pan unit does, starting from a random heading; each beam ends at the first
    obstacle (the reflector) or at max_range. Beams shorter than min_length
    are dropped, so a pose next to a wall may keep fewer beams.

    Returns (beams, pose_ids, poses): beams as [((x0, y0), (x1, y1)), ...] in
    meters, pose_ids (N,) int array, poses (P, 2) array.
    """
    rng = rng or np.random.default_rng()
    if poses is None:
        poses = select_vantage_poses(obstacles, cell_size, num_poses, min_clearance,
                                     max_range, min_angle, coverage)

    # Beams leaving through an opening in the map border end on it; clamping
    # keeps rounding errors from placing that endpoint just outside the map
    map_w, map_h = obstacles.shape[1] * cell_size, obstacles.shape[0] * cell_size
    beams, pose_ids = [], []
    for p, (x0, y0) in enumerate(poses):
        for a in sweep_angles(rng.uniform(0, 2 * np.pi), angular_step, fov):
            d = raycast_grid(obstacles, cell_size, (x0, y0), a, max_range)
            if d < min_length:
                continue
            x1 = min(max(x0 + d * math.cos(a), 0.0), map_w)
            y1 = min(max(y0 + d * math.sin(a), 0.0), map_h)
            beams.append(((float(x0), float(y0)), (float(x1), float(y1))))
            pose_ids.append(p)
    return beams, np.asarray(pose_ids, dtype=np.int64), np.asarray(poses)


def simulate_gas_integrals(
    gas_concentration_map: np.ndarray,
    beams: list,
    cell_dimensions_meters: float,
    quiet: bool = False
) -> np.ndarray:
    """Simulates TDLAS measurements: the integral of the map along each beam.

    Every cell contributes its concentration times the length of the beam
    inside it (exact traversal, see ray_cell_lengths). Beams leaving the map
    measure 0.
    """
    lengths, valid = ray_cell_lengths(beams, gas_concentration_map.shape, cell_dimensions_meters)
    if not quiet:
        for (x0, y0), (x1, y1) in np.asarray(beams, dtype=float).reshape(-1, 2, 2)[~valid]:
            print(f"Warning: Beam ({x0}, {y0}) - ({x1}, {y1}) is out of map boundaries. Skipping.")
    return lengths @ np.asarray(gas_concentration_map, dtype=np.float64).ravel()

# ==========================================
#         GENERATE GROUND TRUTH
# ==========================================

def add_measurement_noise(y_true, snr_db=30, abs_std=0.0, rng: np.random.Generator | None = None):
    """Additive Gaussian noise: relative part set by snr_db, plus an optional
    absolute floor abs_std (same units as y, e.g. ppm*m)."""
    rng = rng or np.random.default_rng()
    y_true_np = y_true.cpu().numpy()
    signal_power = np.mean(y_true_np**2)
    noise_power = signal_power / (10**(snr_db / 10)) + abs_std**2
    noise = rng.normal(0, np.sqrt(noise_power), size=y_true_np.shape)
    y_noisy = y_true_np + noise
    y_noisy[y_noisy < 0] = 0
    return torch.tensor(y_noisy, dtype=torch.float32, device=y_true.device)

def simulation_rngs(seed: int | None) -> tuple[np.random.Generator, ...]:
    """Independent generators for the gas map, the beams and the noise.

    Each one comes from its own stream of the seed, so any of them can change
    (e.g. a different noise level) without altering what the others draw.
    seed=None gives different data on every call.
    """
    return tuple(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))

def generate_simulation_data(cfg: Config) -> tuple[MeasurementBatch, EnvironmentContext]:
    """Generates gas distribution (ground truth), beams and measurements"""
    
    quiet = cfg.quiet
    map_rng, beam_rng, noise_rng = simulation_rngs(cfg.sim.seed)

    if cfg.sim.scenario:
        return generate_scenario_data(cfg, beam_rng, noise_rng)

    # ----- Ground Truth -----
    if not quiet: print("Generating procedural ground truth...")
    map_w, map_h = cfg.env.map_size
    grid_w = round(map_w / cfg.env.cell_size)
    grid_h = round(map_h / cfg.env.cell_size)

    img_gt = generate_fractal_gas_distribution(grid_size=(grid_h, grid_w), rng=map_rng)

    # ------ Beams ------
    if not quiet: print("Generating beams...")
    beams_list = generate_simple_beams(
        cfg.env.map_size,
        cfg.sim.num_beams,
        rng=beam_rng,
    )

    beams_tensor = torch.tensor(beams_list, dtype=torch.float32, device=cfg.device)

    # ------- Measurements --------
    measurements_list = simulate_gas_integrals(img_gt, beams_list, cfg.env.cell_size, quiet=quiet)
    y_true = torch.tensor(measurements_list, dtype=torch.float32, device=cfg.device)

    if cfg.sim.noise:
        if not quiet: print(f"Adding noise to the measurements ({cfg.sim.snr_db} dB)...")
        num_beams = len(beams_list)
        if num_beams > 0:
            measurements = add_measurement_noise(y_true[:num_beams], snr_db=cfg.sim.snr_db, rng=noise_rng)
        else:
            measurements = y_true
    else:
        measurements = y_true

    return (
        MeasurementBatch(beams=beams_tensor, measurements=measurements),
        EnvironmentContext(ground_truth=GroundTruth(gas_map=img_gt, y_true=y_true))
    )

def generate_scenario_data(cfg: Config, beam_rng: np.random.Generator,
                           noise_rng: np.random.Generator) -> tuple[MeasurementBatch, EnvironmentContext]:
    """Measurements on a GADEN scenario package (see gaden_utils.py).

    Uses cfg.sim.scenario (package directory) and the vantage-point fields of
    cfg.sim (num_poses, angular_step, beam_fov, min_clearance, max_range,
    min_crossing_angle, coverage, noise_floor). cfg.sim.num_beams is not used:
    the number of beams follows from the poses the map needs and the angular
    resolution of the sweep. The distribution is quasi-static, so every beam
    integrates the same snapshot, which is also the ground truth.
    """
    from utils.gaden_utils import load_scenario

    quiet = cfg.quiet
    sc = load_scenario(cfg.sim.scenario)
    cfg.env.cell_size = sc.meta["cell_size"]
    cfg.env.map_size = tuple(sc.meta["map_size_m"])
    obstacles = sc.obstacles

    if not quiet: print(f"Loaded scenario {cfg.sim.scenario} "
                        f"({obstacles.shape[1]}x{obstacles.shape[0]} cells). Generating beams...")
    beams_list, pose_ids, poses = generate_vantage_beams(
        obstacles, cfg.env.cell_size, cfg.sim.angular_step,
        num_poses=cfg.sim.num_poses,
        min_clearance=cfg.sim.min_clearance,
        max_range=cfg.sim.max_range,
        min_angle=cfg.sim.min_crossing_angle,
        coverage=cfg.sim.coverage,
        fov=cfg.sim.beam_fov,
        rng=beam_rng,
    )
    if not quiet: print(f"{len(beams_list)} beams from {len(poses)} poses "
                        f"(one every {cfg.sim.angular_step} deg)")

    # Measurements integrate the finer map when the package has one, so that the grid
    # methods do not reconstruct on the very grid the data was generated on
    if sc.gas_map_fine is not None:
        integrals = simulate_gas_integrals(sc.gas_map_fine, beams_list, cfg.env.cell_size / sc.meta["fine_factor"],
                                           quiet=quiet)
    else:
        integrals = simulate_gas_integrals(sc.gas_map, beams_list, cfg.env.cell_size, quiet=quiet)
    dev = cfg.device
    y_true = torch.tensor(integrals, dtype=torch.float32, device=dev)
    measurements = y_true
    if cfg.sim.noise:
        if not quiet: print(f"Adding noise to the measurements ({cfg.sim.snr_db} dB)...")
        measurements = add_measurement_noise(measurements, snr_db=cfg.sim.snr_db,
                                             abs_std=cfg.sim.noise_floor, rng=noise_rng)

    return (
        MeasurementBatch(beams=torch.tensor(beams_list, dtype=torch.float32, device=dev),
                         measurements=measurements,
                         pose_ids=torch.tensor(pose_ids, device=dev)),
        EnvironmentContext(obstacles=obstacles,
                           ground_truth=GroundTruth(gas_map=sc.gas_map, y_true=y_true)),
    )
