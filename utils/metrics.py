"""
Reconstruction metrics of the experiments: the error of a gas map against the ground
truth (gas_map_metrics), also per region of the free space by how the poses observe
it (observation_levels).
"""
import numpy as np

from utils.sim_utils import MeasurementBatch
from utils.geometry_utils import visible_cells

# Ground-truth level, relative to its peak, below which a cell counts as free of gas
CLEAN_LEVEL = 0.01
# Free cells by how the poses observe them (observation_levels)
REGIONS = {"observed": (1, 2), "unobserved": (0,), "covered": (2,), "single": (1,)}


def gas_map_metrics(gt: np.ndarray, pred: np.ndarray, obstacles: np.ndarray | None = None,
                    cell_size: float = 1.0, levels: np.ndarray | None = None) -> dict:
    """Reconstruction metrics, computed on the free cells where gas can be.

    nrmse: RMSE divided by the ground-truth peak, comparable across scenarios.
    peak_ratio: reconstructed peak / true peak.
    peak_distance: distance between the true and the reconstructed maxima, in the
        units of cell_size (meters for scenarios).
    mass_error: relative error of the total amount of gas.
    leak: fraction of the reconstructed gas that lies inside obstacles.

    With the observation levels of the free cells (observation_levels), also per region
    of REGIONS (observed, unobserved, covered from two directions, seen from a single one):
    nrmse_<region>: RMSE over the region's cells divided by the same peak (NaN if empty);
        the squares weighted by the area shares add up to nrmse^2.
    area_<region>: share of the free cells in the region.
    true_gas_<region>, gas_<region>: share of the true / reconstructed gas in free space
        that lies in the region.
    false_gas_unobserved: share of the reconstructed gas in free space that lies in
        unobserved cells where the ground truth is below CLEAN_LEVEL of its peak.
    The unobserved cells are also split by the ground truth into two regions with the
    same nrmse_<region> and area_<region>: "false" (no true gas, below CLEAN_LEVEL of the
    peak, so the error there is gas the method placed where there is none) and "missing"
    (true gas, so the error there is mostly gas the method did not place).
    """
    free = np.ones(gt.shape, dtype=bool) if obstacles is None else ~(obstacles > 0.5)
    g, p = gt[free], pred[free]
    peak = max(float(g.max()), 1e-12)
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    true_max = np.unravel_index(np.argmax(np.where(free, gt, -np.inf)), gt.shape)
    pred_max = np.unravel_index(np.argmax(np.where(free, pred, -np.inf)), gt.shape)
    metrics = {
        "rmse": rmse,
        "nrmse": rmse / peak,
        "peak_ratio": float(p.max()) / peak,
        "peak_distance": float(np.hypot(true_max[0] - pred_max[0], true_max[1] - pred_max[1])) * cell_size,
        "mass_error": float(p.sum() - g.sum()) / max(float(g.sum()), 1e-12),
        "leak": float(pred[~free].sum()) / max(float(pred.sum()), 1e-12),
    }
    if levels is not None:
        lv = levels[free]
        g_mass, p_mass = max(float(g.sum()), 1e-12), max(float(p.sum()), 1e-12)
        for name, members in REGIONS.items():
            region = np.isin(lv, members)
            metrics[f"nrmse_{name}"] = (float(np.sqrt(np.mean((p[region] - g[region]) ** 2))) / peak
                                        if region.any() else float("nan"))
            metrics[f"area_{name}"] = float(region.mean())
            metrics[f"true_gas_{name}"] = float(g[region].sum()) / g_mass
            metrics[f"gas_{name}"] = float(p[region].sum()) / p_mass
        metrics["false_gas_unobserved"] = float(p[(lv == 0) & (g < CLEAN_LEVEL * peak)].sum()) / p_mass
        has_gas = g >= CLEAN_LEVEL * peak
        for name, region in (("false", (lv == 0) & ~has_gas), ("missing", (lv == 0) & has_gas)):
            metrics[f"nrmse_{name}"] = (float(np.sqrt(np.mean((p[region] - g[region]) ** 2))) / peak
                                        if region.any() else float("nan"))
            metrics[f"area_{name}"] = float(region.mean())
    return metrics


def pose_positions(batch: MeasurementBatch) -> np.ndarray:
    """Vantage pose of every pose id, (P, 2) in meters: the common origin of its beams."""
    beams, ids = batch.beams.cpu().numpy(), batch.pose_ids.cpu().numpy()
    return np.array([beams[ids == p][0, 0] for p in np.unique(ids)])


def observation_levels(obstacles: np.ndarray, cell_size: float, poses: np.ndarray, max_range: float,
                       min_angle: float, num_rays: int = 2048) -> np.ndarray:
    """How each cell is observed from the poses, (H, W): 0 no pose has a line of sight to
    it, 1 seen along a single direction, 2 seen from two poses along lines at least
    min_angle degrees apart (the rule select_vantage_poses covers the free space with).

    Line of sight is traced with num_rays rays per pose (visible_cells), independently of
    the angular step of the sweep: a cell between two beams of a pose still counts as seen.
    2048 rays are 0.18 deg apart, less than a 0.1 m cell up to 30 m away.
    """
    H, W = obstacles.shape
    visible = visible_cells(poses, obstacles > 0.5, cell_size, max_range, num_rays)   # (P, H*W)
    rows, cols = np.divmod(np.arange(H * W), W)
    xy = np.stack([(cols + 0.5) * cell_size, (rows + 0.5) * cell_size], axis=1)
    # direction of the line from each pose to each cell, in [0, pi)
    bearing = np.mod(np.arctan2(xy[None, :, 1] - poses[:, None, 1], xy[None, :, 0] - poses[:, None, 0]), np.pi)
    covered = np.zeros(H * W, dtype=bool)
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            diff = np.abs(bearing[i] - bearing[j])
            covered |= visible[i] & visible[j] & (np.minimum(diff, np.pi - diff) >= np.radians(min_angle))
    return (visible.any(axis=0).astype(np.int8) + covered).reshape(H, W)
