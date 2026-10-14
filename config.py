import torch

from typing import Optional, Tuple
from dataclasses import dataclass
from simple_parsing import field

@dataclass
class EnvironmentParams:
    map_size: Tuple[float, float] = (20.0, 20.0) # Map size (map_width, map_height) in meters. Ignored when a csv file is provided
    cell_size: float = 1.0 # Cell size in meters

@dataclass
class SimulationParams:
    seed: Optional[int] = 0 # Seed for the simulated data (gas map, beams, noise). None: different data on every run
    num_beams: int = 100 # Total number of TDLAS beams (procedural maps; scenarios use angular_step)
    
    noise: bool = field(default=False, action="store_true") # Add noise to simulated measurements
    snr_db: int = 30 # Signal-to-noise ratio (dB)
    noise_floor: float = 0.0 # Absolute noise std added on top of the SNR noise (ppm*m)

    # --- GADEN scenarios (see utils/gaden_utils.py) ---
    scenario: Optional[str] = None # Scenario package directory. Overrides map/cell size. If None, a procedural map is generated
    num_poses: Optional[int] = None # Robot vantage poses. None: as many as the coverage target needs
    angular_step: float = 2.0 # Angle between consecutive beams of the sweep at each pose (deg)
    min_clearance: float = 0.3 # Minimum distance from a pose to any obstacle (m)
    max_range: float = 30.0 # Maximum beam length (m)
    min_crossing_angle: float = 30.0 # A free cell is covered once beams from two poses cross it at least this angle apart (deg)
    coverage: float = 0.95 # Fraction of free cells to cover when num_poses is None
    beam_fov: float = 360.0 # Angular sweep at each pose (deg)

@dataclass
class TrainParams:
    pos_lr: float = 0.008 # Position learning rate
    scale_lr: float = 0.003 # Scale learning rate
    rotation_lr: float = 0.001 # Rotation learning rate
    concentration_lr: float = 0.005 # Concentration learning rate

    iterations: int = 1500 # Max number of iterations

    scale_update_factor: float = 2.0 # Raise the concentration scale when a batch exceeds it by this factor

    obstacle_lambda: float = 0.1 # Weight for the obstacle penalty term in the loss function
    obstacle_sigma_k: float = 2.0 # Margin kept between a Gaussian and an obstacle, in standard deviations along the evaluated direction

    early_stopping_patience: int = 100      # How many iterations to wait for an improvement
    early_stopping_min_delta: float = 1e-3  # Minimum improvement required to reset the patience counter
    ema_alpha: float = 0.6                  # Smoothing factor (Lower = smoother, more memory of past loss)

    do_eval: bool = field(default=False, action="store_true") # Evaluate model during training using ground truth
    eval_interval: int = 25 # Model evaluation interval

    live_vis: bool = field(default=False, action="store_true") # Visualize training progress in real-time

@dataclass
class DensificationParams:
    gradient_threshold: float = 0.001 # Threshold for gradient-based densification
    scale_threshold: float = 0.05 # Threshold used by original densification method. It decides whether the Gaussian should be splitted or cloned
    prune_threshold: float = 0.005 # Prune Gaussians below this concentration (relative to the concentration scale)
    injection_concentration: float = 0.01 # Concentration of injected Gaussians (relative to the concentration scale)
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

    # "cuda" if available, "cpu" otherwise. Can be overwritten
    device_type: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> torch.device:
        if self.device_type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
        return torch.device(self.device_type)

@dataclass
class ExperimentConfig(Config):
    num_seeds: int = 30 # Number of random seeds for the experiment