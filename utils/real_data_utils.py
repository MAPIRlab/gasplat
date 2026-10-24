"""
Real TDLAS measurements as GASPLAT scenario packages.

A real scenario package is the counterpart of the GADEN packages of
gaden_utils.py for measurements taken with a real sensor, and main.py loads it
the same way, with --scenario. There is no ground truth and no occupancy map:
the reconstruction can only be inspected, not evaluated. The packages live in
real_data/ (e.g. real_data/full_sweep):
    measurements.npz   beams        (N, 2, 2) float32  sensor and reflector in the local frame [m]
                       measurements (N,) float32       beam integrals projected onto the plane [ppm*m]
                       pose_ids     (N,) int64         vantage pose of each beam
    meta.json          cell size, map size, world origin of the local frame, ...

Conventions as in gaden_utils.py: the map covers [0, W*cs] x [0, H*cs] and
meta["origin_world"] is the world coordinate of local (0, 0). The map is the
bounding box of the beams plus a margin.
"""
import os
import json
import torch
import numpy as np

from config import Config
from utils.sim_utils import MeasurementBatch, EnvironmentContext


def is_real_scenario(path: str) -> bool:
    return os.path.exists(os.path.join(path, "measurements.npz"))


def load_real_scenario(cfg: Config) -> tuple[MeasurementBatch, EnvironmentContext]:
    """Measurements of a real scenario package (cfg.sim.scenario), with no ground truth or obstacles.

    Sets the map and cell size of cfg from the package. cfg.sim.num_poses keeps
    the first poses only; the other options of cfg.sim do not apply.
    """
    data = np.load(os.path.join(cfg.sim.scenario, "measurements.npz"))
    with open(os.path.join(cfg.sim.scenario, "meta.json")) as f:
        meta = json.load(f)
    cfg.env.cell_size = meta["cell_size"]
    cfg.env.map_size = tuple(meta["map_size_m"])

    keep = np.ones(len(data["pose_ids"]), dtype=bool)
    if cfg.sim.num_poses is not None:
        keep = data["pose_ids"] < cfg.sim.num_poses
    dev = cfg.device
    batch = MeasurementBatch(beams=torch.tensor(data["beams"][keep], device=dev),
                             measurements=torch.tensor(data["measurements"][keep], device=dev),
                             pose_ids=torch.tensor(data["pose_ids"][keep], device=dev))
    if not cfg.quiet:
        print(f"Loaded real scenario {cfg.sim.scenario}: {len(batch.measurements)} beams "
              f"from {int(batch.pose_ids.max()) + 1} poses")
    return batch, EnvironmentContext()
