from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from gs_model import GasSplattingModel
    from utils.sim_utils import EnvironmentContext


def add_rectangle(
    obstacles: np.ndarray,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> None:
    """Add a rectangle using coordinates normalized to the grid size."""
    grid_h, grid_w = obstacles.shape
    x0 = max(0, min(grid_w, int(x_min * grid_w)))
    x1 = max(0, min(grid_w, int(x_max * grid_w)))
    y0 = max(0, min(grid_h, int(y_min * grid_h)))
    y1 = max(0, min(grid_h, int(y_max * grid_h)))
    obstacles[y0:y1, x0:x1] = 1.0


def add_circle(
    obstacles: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> None:
    """Add a circle using normalized grid coordinates."""
    grid_h, grid_w = obstacles.shape
    y, x = np.indices(obstacles.shape)
    x = x / grid_w
    y = y / grid_h
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    obstacles[mask] = 1.0


def create_obstacle_scenario(name: str, grid_shape: tuple[int, int]) -> np.ndarray:
    """Create a reproducible binary obstacle mask for a named scenario."""
    obstacles = np.zeros(grid_shape, dtype=np.float32)

    if name == "none":
        return obstacles
    if name == "small_square":
        add_rectangle(obstacles, 0.40, 0.40, 0.60, 0.60)
    elif name == "large_square":
        add_rectangle(obstacles, 0.25, 0.25, 0.75, 0.75)
    elif name == "vertical_wall":
        add_rectangle(obstacles, 0.48, 0.10, 0.52, 0.90)
    elif name == "two_blocks":
        add_rectangle(obstacles, 0.15, 0.20, 0.35, 0.80)
        add_rectangle(obstacles, 0.65, 0.20, 0.85, 0.80)
    elif name == "l_shape":
        add_rectangle(obstacles, 0.20, 0.20, 0.30, 0.80)
        add_rectangle(obstacles, 0.20, 0.70, 0.70, 0.80)
    elif name == "u_shape":
        add_rectangle(obstacles, 0.20, 0.20, 0.30, 0.80)
        add_rectangle(obstacles, 0.70, 0.20, 0.80, 0.80)
        add_rectangle(obstacles, 0.20, 0.20, 0.80, 0.30)
    elif name == "circle":
        add_circle(obstacles, 0.50, 0.50, 0.25)
    elif name == "boundary_wall":
        add_rectangle(obstacles, 0.00, 0.35, 0.08, 0.65)
    else:
        raise ValueError(f"Unknown obstacle scenario: {name}")

    return obstacles


def obstacle_mask_to_geometry(
    obstacles: np.ndarray,
    cell_size: float,
):
    """Convert occupied grid cells into one unioned Shapely geometry."""
    polygons = []
    for row, col in zip(*np.where(obstacles > 0.5)):
        x_min, x_max = col * cell_size, (col + 1) * cell_size
        y_min, y_max = row * cell_size, (row + 1) * cell_size
        polygons.append(Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
        ]))
    return unary_union(polygons) if polygons else None



def _steps_to_nearest_marked_cell(
    marked: np.ndarray,
    axis: int,
    unreachable: float,
) -> np.ndarray:
    """Number of cells to the nearest marked cell, either way along one axis.

    Works on cell indices rather than on distances. Replacing every unmarked
    cell by an index that loses any comparison, then taking a running minimum
    along the axis, leaves each cell holding the index of the closest marked
    cell at or after it; running a maximum the other way gives the closest one
    at or before it. Subtracting the cell's own index turns those into counts.

    Marked cells resolve to zero, and cells with no marked cell on a given side
    to `unreachable`, which must be at least as large as any distance the caller
    intends to keep.
    """
    length = marked.shape[axis]
    position = np.arange(length, dtype=np.float32)
    position = position.reshape((-1, 1) if axis == 0 else (1, -1))

    # Index of the closest marked cell at or after each cell, then at or before.
    # The fillers sit just outside the axis, so "none on this side" is explicit
    # rather than a large index that would be mistaken for a real one.
    after = np.flip(np.minimum.accumulate(
        np.flip(np.where(marked, position, length), axis), axis), axis)
    before = np.maximum.accumulate(np.where(marked, position, -1.0), axis)

    return np.minimum(
        np.where(after < length, after - position, unreachable),
        np.where(before >= 0.0, position - before, unreachable),
    )


def compute_esdf(obstacles: np.ndarray, cell_size: float) -> np.ndarray:
    """Euclidean signed distance field (ESDF) D in meters, negative inside obstacles.

    The transform measures between cell centers, so half a cell is taken off to
    refer the distance to the obstacle surface.
    """
    occupied = obstacles > 0.5
    dist_outside = np.asarray(ndimage.distance_transform_edt(~occupied))
    dist_inside = np.asarray(ndimage.distance_transform_edt(occupied))

    cells = dist_outside - dist_inside
    return (np.where(cells > 0, cells - 0.5, cells + 0.5) * cell_size).astype(np.float32)


def compute_esdf_gradient(esdf: np.ndarray) -> np.ndarray:
    """Gradient of the ESDF by finite differences, (2, H, W) as (x, y)."""
    grad_y, grad_x = np.gradient(esdf)
    return np.stack([grad_x, grad_y], axis=0).astype(np.float32)


def compute_aasdf(
    obstacles: np.ndarray,
    cell_size: float,
    max_dist: float,
) -> np.ndarray:
    """Axis-aligned signed distance field (AA-SDF) D_x, D_y in meters.

    Distance to the nearest obstacle along each axis, taking the closer of the
    two senses, negative inside obstacles. Axes with no wall in sight saturate
    at `max_dist`. Returns an array (2, H, W) ordered (x, y).
    """
    occupied = obstacles > 0.5
    free = ~occupied
    grid_h, grid_w = occupied.shape

    # Larger than any reachable step count, so "no wall ahead" saturates below
    unreachable = float(grid_h + grid_w)

    def signed_steps(axis: int) -> np.ndarray:
        """Steps to the nearest wall along one array axis, negative inside it."""
        # Outside, how far to the first obstacle. Inside, how far back out.
        # Cells are sampled at their center, hence the half step offset, which
        # keeps the ramp linear all the way across the wall face.
        return np.where(
            occupied,
            -(_steps_to_nearest_marked_cell(free, axis, unreachable) - 0.5),
            _steps_to_nearest_marked_cell(occupied, axis, unreachable) - 0.5,
        )

    # The x channel comes first so that the penalty can read the variance
    # projected on each axis straight off the diagonal of the covariance matrix
    distances = np.stack([
        signed_steps(axis=1),  # scan across columns: the x axis
        signed_steps(axis=0),  # scan down rows: the y axis
    ]) * cell_size

    return np.clip(distances, -max_dist, max_dist).astype(np.float32)


@dataclass
class DistanceFields:
    """Distance fields used by the obstacle penalty L_obs, all in meters, on the device:
    the ESDF D (1, 1, H, W), its gradient (1, 2, H, W) and the AA-SDF D_x, D_y (1, 2, H, W)."""
    esdf: torch.Tensor
    esdf_grad: torch.Tensor
    aasdf: torch.Tensor

    @classmethod
    def from_environment(cls, environment: EnvironmentContext | None, cfg: Config) -> DistanceFields | None:
        """Fields of the obstacle map of the environment, None without one."""
        if not environment or environment.obstacles is None:
            return None

        esdf = compute_esdf(environment.obstacles, cell_size=cfg.env.cell_size)
        dev = cfg.device
        aasdf = compute_aasdf(
            environment.obstacles,
            cell_size=cfg.env.cell_size,
            max_dist=float(max(cfg.env.map_size)),
        )
        return cls(
            esdf=torch.tensor(esdf, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0),
            esdf_grad=torch.tensor(compute_esdf_gradient(esdf), dtype=torch.float32, device=dev).unsqueeze(0),
            aasdf=torch.tensor(aasdf, dtype=torch.float32, device=dev).unsqueeze(0),
        )


def _sample(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup of a field (1, C, H, W) at the points of a grid (1, 1, K, 2)
    normalized to [-1, 1]: (1, C, 1, K) -> (K, C)."""
    return F.grid_sample(
        field, grid,
        mode='bilinear', padding_mode='border', align_corners=False
    ).squeeze(0).squeeze(1).transpose(0, 1)


def obstacle_penalty(model: GasSplattingModel, fields: DistanceFields, cfg: Config) -> torch.Tensor:
    """Obstacle penalty L_obs: ESDF and AA-SDF terms, complementary. A Gaussian is
    penalized when the obstacle surface lies closer than obstacle_sigma_k standard
    deviations, measured along the normal to the nearest surface (ESDF) and along
    each coordinate axis (AA-SDF)."""
    pos = model.get_pos()                      # (K, 2)
    Sigma = model.get_covariance()             # (K, 2, 2)
    concentrations = model.get_concentration() # (K,)

    # Normalize positions to [-1, 1] for grid sampling
    map_w, map_h = model.map_size[0], model.map_size[1]
    norm_x = (pos[:, 0] / map_w) * 2.0 - 1.0
    norm_y = (pos[:, 1] / map_h) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2) # (1, 1, K, 2)

    # --- ESDF term ---
    esdf_at_mu = _sample(fields.esdf, grid).squeeze(1)  # D(mu_k): (K,)
    esdf_grad_at_mu = _sample(fields.esdf_grad, grid)   # grad D(mu_k): (K, 2)

    # Unit normal n_k pointing away from the nearest surface
    n = F.normalize(esdf_grad_at_mu, p=2, dim=-1, eps=1e-6) # (K, 2)

    # Variance along the normal, n^T Sigma n: (K, 1, 2) @ (K, 2, 2) @ (K, 2, 1) -> (K,)
    normal_variance = torch.bmm(
        n.unsqueeze(1), torch.bmm(Sigma, n.unsqueeze(-1))
    ).view(-1)
    sigma_n = torch.sqrt(torch.clamp(normal_variance, min=1e-7))

    esdf_violation = F.relu(cfg.train.obstacle_sigma_k * sigma_n - esdf_at_mu)

    # --- AA-SDF term ---
    aasdf_at_mu = _sample(fields.aasdf, grid)  # (D_x(mu_k), D_y(mu_k)): (K, 2)

    # Along the coordinate axes the variance reduces to the diagonal of Sigma: (K, 2)
    sigma_axis = torch.sqrt(torch.clamp(torch.diagonal(Sigma, dim1=1, dim2=2), min=1e-7))

    aasdf_violation = F.relu(cfg.train.obstacle_sigma_k * sigma_axis - aasdf_at_mu)

    # Weight by sg(rho_k): keeps the penalty commensurate with the data
    # term, and stops fading a Gaussian out from dodging it.
    return cfg.train.obstacle_lambda * torch.sum(
        (esdf_violation + aasdf_violation.sum(dim=-1)) * concentrations.detach()
    ) / model.scale
