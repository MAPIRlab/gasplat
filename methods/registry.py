import time
from scipy.sparse import csr_matrix

import methods.tomography as tm
from utils.sim_utils import EnvironmentContext, MeasurementBatch
from config import Config
from trainer import fit_gasplat

def get_tomo_inputs(batch: MeasurementBatch, cfg: Config):
    """Helper to prevent repeating grid math and tensor conversions."""
    grid_w = round(cfg.env.map_size[0] / cfg.env.cell_size)
    grid_h = round(cfg.env.map_size[1] / cfg.env.cell_size)
    measurements = batch.measurements.cpu().numpy()
    return (grid_h, grid_w), measurements

def timed(func, *args, **kwargs):
    """Runs func(*args, **kwargs), returns (result, time)."""
    t0 = time.time()
    res = func(*args, **kwargs)
    return res, time.time() - t0

# Every method returns (reconstruction, time). Grid methods receive the system
# matrix already built, so their time is the solve alone. Their tuned parameters
# are the defaults of methods/tomography.py.

def run_sart(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)
    return timed(tm.sart, system_matrix, measurements, grid_size=grid_size, quiet=cfg.quiet)

def run_rbf_sart(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)
    return timed(tm.rbf_sart, system_matrix, measurements, grid_size=grid_size, cell_size_m=cfg.env.cell_size,
                 quiet=cfg.quiet)

def run_lfd(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)
    return timed(tm.lfd, system_matrix, measurements, grid_size=grid_size)

def run_ltd(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)
    return timed(tm.ltd, system_matrix, measurements, grid_size=grid_size)

def run_gas_splatting(batch: MeasurementBatch, cfg: Config, environment: EnvironmentContext | None = None, **kwargs):
    model, _, fit_time = fit_gasplat(batch, cfg, environment)
    return model.render_map(cell_size=cfg.env.cell_size), fit_time

AVAILABLE_METHODS = {
    "SART": {
        "func": run_sart,
        "needs_matrix": True,
        "style": {"color": "tab:orange", "marker": "o"}
    },
    "RBF Coupled SART": {
        "func": run_rbf_sart,
        "needs_matrix": True,
        "style": {"color": "tab:green", "marker": "s"}
    },
    "LFD": {
        "func": run_lfd,
        "needs_matrix": True,
        "style": {"color": "tab:red", "marker": "^"},
    },
    "LTD": {
        "func": run_ltd,
        "needs_matrix": True,
        "style": {"color": "tab:purple", "marker": "v"}
    },
    "Gas Splatting": {
        "func": run_gas_splatting,
        "style": {"color": "tab:blue", "marker": "*", "linewidth": 3, "markersize": 12}
    }
}