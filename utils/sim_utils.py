import os
import json
import math
import hashlib
import torch
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

from config import Config
from utils.geometry_utils import ray_cell_lengths, raycast_grid, visible_cells


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
    """Generates beams fanning out from both bottom corners, evenly spread in angle over 90 degrees."""
    map_x, map_y = map_size_m

    def fan(n):   # endpoints of the fan from (0, 0)
        ends = []
        for angle in np.linspace(0, np.pi / 2, n):
            if angle <= math.atan2(map_y, map_x):
                ends.append((map_x, map_x * np.tan(angle)))
            else:   # exact 0 at 90 degrees, where 1 / tan is not
                ends.append((0.0 if angle == np.pi / 2 else map_y / np.tan(angle), map_y))
        return ends

    # The right fan mirrors the left one
    left = [((0.0, 0.0), end) for end in fan(num_beams // 2)]
    right = [((map_x, 0.0), (map_x - x1, y1)) for x1, y1 in fan(num_beams - num_beams // 2)]
    return left + right

def generate_random_beams(map_size_m: tuple, num_beams: int, rng: np.random.Generator | None = None):
    """Generates random TDLAS beams between two different edges of a map in meters."""
    rng = rng or np.random.default_rng()
    map_x, map_y = map_size_m
    edges = ['left', 'right', 'bottom', 'top']

    def point_on(edge):
        if edge in ('left', 'right'):
            return (0.0 if edge == 'left' else map_x), rng.uniform(0, map_y)
        return rng.uniform(0, map_x), (0.0 if edge == 'bottom' else map_y)

    beams = []
    for _ in range(num_beams):
        start_edge = rng.choice(edges)
        start = point_on(start_edge)
        beams.append((start, point_on(rng.choice([e for e in edges if e != start_edge]))))
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
    """Beams across a map without obstacles: a third each of random, radial and
    horizontal/vertical beams (the first ones get the remainder)."""
    counts = [num_beams // 3 + (i < num_beams % 3) for i in range(3)]
    return (generate_random_beams(map_size_m, counts[0], rng=rng)
            + generate_radial_beams(map_size_m, counts[1])
            + generate_horizontal_vertical_beams(map_size_m, counts[2]))


# ==========================================
#          VANTAGE-POINT BEAMS
# ==========================================

def vantage_candidates(obstacles: np.ndarray, cell_size: float, min_clearance: float = 0.3,
                       spacing: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Positions where the robot can stop: free cells on a grid of `spacing` m with at
    least `min_clearance` m to any obstacle, in the largest connected free region.

    Returns the (C, 2) positions in meters and the (H, W) mask of that region.
    """
    from scipy.ndimage import distance_transform_edt, label

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
    return np.stack([(cols + 0.5) * cell_size, (rows + 0.5) * cell_size], axis=1), reachable


def random_vantage_poses(obstacles: np.ndarray, cell_size: float, num_poses: int,
                         min_clearance: float = 0.3, min_spacing: float = 1.0,
                         rng: np.random.Generator | None = None, tries: int = 100) -> np.ndarray:
    """num_poses positions drawn at random among the candidates of vantage_candidates,
    at least `min_spacing` m apart.

    The candidates are visited in a random order and each one is kept if it is far
    enough from those already kept; if fewer than num_poses fit, a new order is drawn.
    Returns (num_poses, 2) positions in meters.
    """
    rng = rng or np.random.default_rng()
    cand, _ = vantage_candidates(obstacles, cell_size, min_clearance)
    for _ in range(tries):
        chosen = []
        for i in rng.permutation(len(cand)):
            if all(np.hypot(*(cand[i] - cand[j])) >= min_spacing for j in chosen):
                chosen.append(i)
                if len(chosen) == num_poses:
                    return cand[chosen]
    raise ValueError(f"Could not place {num_poses} poses {min_spacing} m apart.")


def select_vantage_poses(obstacles: np.ndarray, cell_size: float,
                         num_poses: int | None = None, min_clearance: float = 0.3,
                         max_range: float = 30.0, min_angle: float = 30.0,
                         coverage: float = 0.8, spacing: float = 0.3) -> np.ndarray:
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
    seen from a single direction and 2 for a covered one. If num_poses is None,
    selection stops once a fraction `coverage` of the free cells is covered (or
    no candidate improves it further). Otherwise it takes the first num_poses
    of a fixed order: the same steps until no candidate improves the coverage,
    then the candidate that sees the most cells along a line at least
    `min_angle` from every line they are already seen along, so that each pose
    adds the most new viewing directions. The poses for any coverage target or
    smaller num_poses are a prefix of that order.

    Returns (P, 2) positions in meters.
    """
    H, W = obstacles.shape
    cand, reachable = vantage_candidates(obstacles, cell_size, min_clearance, spacing)

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

    def far(p):   # lines from candidates to each cell at least min_angle from the line from pose p
        diff = np.abs(bearing - bearing[p])
        return np.minimum(diff, np.pi - diff) >= min_angle

    while num_poses is None or len(chosen) < num_poses:
        gain = (vis & (level == 0)).sum(axis=1) + (vis & crosses & (level == 1)).sum(axis=1)
        best = int(np.argmax(gain))
        if gain[best] == 0:
            break
        chosen.append(best)
        seen = vis[best]
        level[seen & (level == 0)] = 1
        level[seen & crosses[best]] = 2
        crosses |= seen & far(best)
        if num_poses is None and (level == 2).sum() >= coverage * num_target:
            break
    if num_poses is None or len(chosen) == num_poses:
        return cand[chosen]

    # new[c, i]: candidate c sees cell i along a line at least min_angle from every chosen pose that sees it
    new = vis.copy()
    for p in chosen:
        new &= ~vis[p] | far(p)
    while len(chosen) < num_poses:
        gain = new.sum(axis=1)
        best = int(np.argmax(gain))
        if gain[best] == 0:
            raise ValueError(f"Only {len(chosen)} poses add viewing directions, {num_poses} requested.")
        chosen.append(best)
        new &= ~vis[best] | far(best)
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


def cached_vantage_poses(cache_dir: str, obstacles: np.ndarray, cell_size: float, num_poses: int | None,
                         min_clearance: float, max_range: float, min_angle: float, coverage: float) -> np.ndarray:
    """select_vantage_poses, stored in cache_dir. The poses depend only on the map and
    these settings, and choosing them is the slowest part of generating the data."""
    settings = json.dumps([num_poses, min_clearance, max_range, min_angle, coverage, cell_size])
    key = hashlib.sha1(settings.encode() + np.ascontiguousarray(obstacles, dtype=bool).tobytes()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"poses_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    poses = select_vantage_poses(obstacles, cell_size, num_poses, min_clearance, max_range, min_angle, coverage)
    tmp = f"{path}.{os.getpid()}.tmp"  # parallel runs may write the same file
    with open(tmp, "wb") as f:
        np.save(f, poses)
    os.replace(tmp, path)
    return poses


def generate_vantage_beams(obstacles: np.ndarray, cell_size: float, angular_step: float,
                           num_poses: int | None = None, poses: np.ndarray | None = None,
                           min_clearance: float = 0.3, max_range: float = 30.0,
                           min_angle: float = 30.0, coverage: float = 0.8,
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
#         MEASUREMENTS OF A CASE
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
    """Measurements of the case that cfg describes, and its environment.

    The case depends on cfg.sim.scenario:
      - None: a procedural map (generate_procedural_data).
      - a GADEN scenario package: beams from vantage poses on its
        snapshot (generate_scenario_data).
      - a real scenario package: the recorded measurements, without ground
        truth or obstacles (real_data_utils.load_real_scenario).
    Packages set cfg.env.map_size and cfg.env.cell_size to their own.

    Simulated measurements are exact integrals of the ground truth, plus noise
    if cfg.sim.noise (cfg.sim.snr_db and cfg.sim.noise_floor); the ground truth
    keeps the noise-free ones (y_true). The gas map, beams and noise draw from
    independent streams of cfg.sim.seed (simulation_rngs).
    """
    from utils.real_data_utils import is_real_scenario, load_real_scenario

    if cfg.sim.scenario and is_real_scenario(cfg.sim.scenario):
        return load_real_scenario(cfg)

    map_rng, beam_rng, noise_rng = simulation_rngs(cfg.sim.seed)
    if cfg.sim.scenario:
        batch, environment = generate_scenario_data(cfg, beam_rng)
    else:
        batch, environment = generate_procedural_data(cfg, map_rng, beam_rng)

    if cfg.sim.noise and len(batch.measurements) > 0:
        if not cfg.quiet: print(f"Adding noise to the measurements ({cfg.sim.snr_db} dB)...")
        batch.measurements = add_measurement_noise(batch.measurements, snr_db=cfg.sim.snr_db,
                                                   abs_std=cfg.sim.noise_floor, rng=noise_rng)
    return batch, environment

def generate_procedural_data(cfg: Config, map_rng: np.random.Generator,
                             beam_rng: np.random.Generator) -> tuple[MeasurementBatch, EnvironmentContext]:
    """Noise-free measurements on a procedural map: fractal gas without obstacles,
    of size cfg.env.map_size and cell cfg.env.cell_size, crossed by
    cfg.sim.num_beams beams (generate_simple_beams)."""
    quiet = cfg.quiet
    if not quiet: print("Generating procedural ground truth...")
    grid_w = round(cfg.env.map_size[0] / cfg.env.cell_size)
    grid_h = round(cfg.env.map_size[1] / cfg.env.cell_size)
    gas_map = generate_fractal_gas_distribution(grid_size=(grid_h, grid_w), rng=map_rng)

    if not quiet: print("Generating beams...")
    beams_list = generate_simple_beams(cfg.env.map_size, cfg.sim.num_beams, rng=beam_rng)
    integrals = simulate_gas_integrals(gas_map, beams_list, cfg.env.cell_size, quiet=quiet)
    y_true = torch.tensor(integrals, dtype=torch.float32, device=cfg.device)

    return (
        MeasurementBatch(beams=torch.tensor(beams_list, dtype=torch.float32, device=cfg.device),
                         measurements=y_true),
        EnvironmentContext(ground_truth=GroundTruth(gas_map=gas_map, y_true=y_true)),
    )

def generate_scenario_data(cfg: Config, beam_rng: np.random.Generator) -> tuple[MeasurementBatch, EnvironmentContext]:
    """Noise-free measurements on a GADEN scenario package (see gaden_utils.py).

    Uses cfg.sim.scenario (package directory) and the vantage-point fields of
    cfg.sim (num_poses, angular_step, beam_fov, min_clearance, max_range,
    min_crossing_angle, coverage, random_poses, min_pose_spacing). With
    random_poses, the poses are drawn at random (random_vantage_poses), as many
    as select_vantage_poses would place. cfg.sim.num_beams is not used: the number
    of beams follows from the poses the map needs and the angular resolution
    of the sweep. The distribution is quasi-static, so every beam integrates
    the same snapshot, which is also the ground truth.
    """
    from utils.gaden_utils import load_scenario

    quiet = cfg.quiet
    sc = load_scenario(cfg.sim.scenario)
    cfg.env.cell_size = sc.meta["cell_size"]
    cfg.env.map_size = tuple(sc.meta["map_size_m"])
    obstacles = sc.obstacles

    if not quiet: print(f"Loaded scenario {cfg.sim.scenario} "
                        f"({obstacles.shape[1]}x{obstacles.shape[0]} cells). Generating beams...")
    poses = cached_vantage_poses(cfg.sim.scenario, obstacles, cfg.env.cell_size, cfg.sim.num_poses,
                                 cfg.sim.min_clearance, cfg.sim.max_range, cfg.sim.min_crossing_angle,
                                 cfg.sim.coverage)
    if cfg.sim.random_poses:
        # As many poses as the coverage target (or num_poses) asks for, placed at random
        poses = random_vantage_poses(obstacles, cfg.env.cell_size, len(poses), cfg.sim.min_clearance,
                                     cfg.sim.min_pose_spacing, rng=beam_rng)
    beams_list, pose_ids, poses = generate_vantage_beams(
        obstacles, cfg.env.cell_size, cfg.sim.angular_step, poses=poses,
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

    return (
        MeasurementBatch(beams=torch.tensor(beams_list, dtype=torch.float32, device=dev),
                         measurements=y_true,
                         pose_ids=torch.tensor(pose_ids, device=dev)),
        EnvironmentContext(obstacles=obstacles,
                           ground_truth=GroundTruth(gas_map=sc.gas_map, y_true=y_true)),
    )
