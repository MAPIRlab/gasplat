import torch

import dataclasses
from typing import Literal, Optional, Tuple
from dataclasses import dataclass
from simple_parsing import field

@dataclass
class EnvironmentParams:
    map_size: Tuple[float, float] = (20.0, 20.0) # Map size (map_width, map_height) in meters, of procedural maps. Scenario packages set their own
    cell_size: float = 1.0 # Cell size in meters, of procedural maps. Scenario packages set their own

@dataclass
class SimulationParams:
    """Simulated measurements. The fields from scenario on apply to GADEN scenarios (see utils/gaden_utils.py)"""
    seed: Optional[int] = 0 # Seed for the simulated data (gas map, beams, noise). None: different data on every run
    num_beams: int = 100 # Total number of TDLAS beams (procedural maps; scenarios use angular_step)
    
    noise: bool = field(default=False, action="store_true") # Add noise to simulated measurements
    snr_db: int = 30 # Signal-to-noise ratio (dB)
    noise_floor: float = 0.0 # Absolute noise std added on top of the SNR noise (ppm*m)
    noise_model: Literal["additive", "proportional"] = "additive" # additive: same std for every beam (snr_db, noise_floor); proportional: std = noise_level * each beam's value
    noise_level: float = 0.05 # Relative std of the proportional noise

    scenario: Optional[str] = None # Scenario package directory. Overrides map/cell size. If None, a procedural map is generated
    num_poses: Optional[int] = None # Robot vantage poses. None: as many as the coverage target needs
    angular_step: float = 6.0 # Angle between consecutive beams of the sweep at each pose (deg)
    min_clearance: float = 0.3 # Minimum distance from a pose to any obstacle (m)
    max_range: float = 30.0 # Maximum beam length (m)
    min_crossing_angle: float = 30.0 # A free cell is covered once beams from two poses cross it at least this angle apart (deg)
    coverage: float = 0.8 # Fraction of free cells to cover when num_poses is None
    beam_fov: float = 360.0 # Angular sweep at each pose (deg)
    random_poses: bool = field(default=False, action="store_true") # Place the poses at random instead (as many as num_poses or the coverage target asks for)
    min_pose_spacing: float = 1.0 # Minimum distance between random poses (m)

@dataclass
class TrainParams:
    pos_lr: float = 0.01 # Position learning rate (meters per step)
    scale_lr: float = 0.024 # Scale learning rate
    rotation_lr: float = 0.001 # Rotation learning rate
    concentration_lr: float = 0.02 # Concentration learning rate

    iterations: int = 1500 # Max number of iterations
    gaussian_support: float = 3.0 # Gaussians are truncated at this many standard deviations: farther beams neither see nor cross them

    obstacle_lambda: float = 0.1 # Weight for the obstacle penalty term in the loss function
    obstacle_sigma_k: float = 1.0 # Margin kept between a Gaussian and an obstacle, in standard deviations along the evaluated direction

    early_stopping_patience: int = 50       # How many iterations to wait for an improvement
    early_stopping_min_delta: float = 2e-2  # Minimum relative improvement of the smoothed loss that resets the patience counter
    ema_alpha: float = 0.6                  # Smoothing factor (Lower = smoother, more memory of past loss)
    known_noise_level: Optional[float] = None   # Known relative std of the measurement noise (fraction of each reading). If set, training stops once the smoothed data residual reaches it (discrepancy principle)

    do_eval: bool = field(default=False, action="store_true") # Evaluate model during training using ground truth
    eval_interval: int = 25 # Model evaluation interval

    live_vis: bool = field(default=False, action="store_true") # Visualize training progress in real-time

@dataclass
class DensificationParams:
    gradient_threshold: float = 0.04 # Densify Gaussians whose position gradient (per meter), averaged over the beams that cross them, exceeds this
    scale_threshold: float = 0.05 # Threshold used by original densification method. It decides whether the Gaussian should be splitted or cloned
    prune_threshold: float = 0.005 # Prune Gaussians below this concentration (relative to the concentration scale)
    injection_concentration: float = 0.01 # Concentration of injected Gaussians (relative to the concentration scale)
    injection_scale: Optional[float] = 0.5 # Std of injected Gaussians and minimum spacing between them (m). None: 10%% of the shorter map side
    densify_from: int = 100 # Densification will start at this iteration
    densify_until: int = 750 # Densification will stop at this iteration
    densify_interval: int = 50 # Iteration interval for densification
    original_dens: bool = field(default=False, action="store_true") # Use the original densification method instead of the proposed one based on Long-Axis Split approach

@dataclass
class Config:
    env: EnvironmentParams
    sim: SimulationParams
    train: TrainParams
    densify: DensificationParams

    # Global flag to silence console output
    quiet: bool = field(default=False, action="store_true")

    # GASPLAT's many small operations run faster on a few CPU threads than on a GPU,
    # where launching them dominates (tuning set: 3.6 s per house with 4 threads, 5.5 s
    # on the GPU). "cuda" to use the GPU
    device_type: str = "cpu"

    @property
    def device(self) -> torch.device:
        if self.device_type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
        return torch.device(self.device_type)


# Fields renamed since older result files were saved: {(group, old name): new name}
RENAMED_FIELDS = {("train", "noise_level"): "known_noise_level"}


def config_from_dict(d: dict) -> Config:
    """Configuration saved with an experiment (dataclasses.asdict of a Config, as in the
    "config" of the result files). Renamed fields of older files (RENAMED_FIELDS) are
    read under their new name; fields this version does not have are ignored, and
    missing ones keep their default."""
    def build(cls, values: dict, group: str | None = None):
        known = {f.name: f for f in dataclasses.fields(cls)}
        kwargs = {}
        for name, value in values.items():
            name = RENAMED_FIELDS.get((group, name), name)
            if name not in known:
                continue
            f = known[name]
            if dataclasses.is_dataclass(f.type):
                value = build(f.type, value, name)
            elif isinstance(value, list):   # JSON keeps tuples as lists
                value = tuple(value)
            kwargs[name] = value
        # The parameter groups have no defaults of their own
        for name, f in known.items():
            if dataclasses.is_dataclass(f.type) and name not in kwargs:
                kwargs[name] = f.type()
        return cls(**kwargs)
    return build(Config, d)
