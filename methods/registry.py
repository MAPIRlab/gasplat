import time
from scipy.sparse import csr_matrix

import methods.tomography as tm
from utils.init_utils import setup_gs_model
from utils.sim_utils import EnvironmentContext, MeasurementBatch
from config import Config
from trainer import Trainer

def get_tomo_inputs(batch: MeasurementBatch, cfg: Config):
    """Helper to prevent repeating grid math and tensor conversions."""
    grid_w = round(cfg.env.map_size[0] / cfg.env.cell_size)
    grid_h = round(cfg.env.map_size[1] / cfg.env.cell_size)
    measurements = batch.measurements.cpu().numpy()
    return (grid_h, grid_w), measurements

# Every method returns (reconstruction, time). Grid methods receive the system
# matrix already built, so their time is the solve alone.

def run_sart(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)

    t0 = time.time()
    res = tm.sart(system_matrix, measurements, grid_size=grid_size, num_iterations=400, relaxation_factor=1.6, quiet=cfg.quiet)
    recon_time = time.time() - t0
    return res, recon_time

def run_rbf_sart(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)

    t0 = time.time()
    res = tm.rbf_sart(system_matrix, measurements, grid_size=grid_size, cell_size_m=cfg.env.cell_size, quiet=cfg.quiet)
    recon_time = time.time() - t0
    return res, recon_time

def run_lfd(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)

    t0 = time.time()
    res = tm.lfd(system_matrix, measurements, grid_size=grid_size, alpha=0.07)
    recon_time = time.time() - t0
    return res, recon_time

def run_ltd(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)
    
    t0 = time.time()
    res = tm.ltd(system_matrix, measurements, grid_size=grid_size, alpha=5.0)
    recon_time = time.time() - t0
    return res, recon_time

def run_gas_splatting(batch: MeasurementBatch, cfg: Config, environment: EnvironmentContext | None = None, **kwargs):
    model, init_data = setup_gs_model(batch, cfg)
    # The trainer precomputes the distance fields of the obstacle penalty; timed like the
    # system matrix of the grid methods
    t0 = time.time()
    trainer = Trainer(model, cfg, environment=environment)
    trainer_setup_time = time.time() - t0
    trainer.train(batch)
    results = trainer.finish()
    gs_img = model.render_map(cell_size=cfg.env.cell_size)

    # Setup and optimization time, as measured by the initialization and the trainer
    return gs_img, init_data.info["setup_time"] + trainer_setup_time + results.training_time

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