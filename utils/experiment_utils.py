import os
import copy
import json
import time
import datetime
import multiprocessing
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.sim_utils import generate_simulation_data
from methods.tomography import create_system_matrix_sparse
from methods.registry import run_gas_splatting

def save_experiment_results(metadata, results, folder="results"):
    """
    Saves metadata and results data in a JSON file.
    """
    os.makedirs(folder, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = metadata.get("experiment_name", "exp")
    filename = f"{exp_name}_{timestamp}.json"
    filepath = os.path.join(folder, filename)
    
    data_to_save = {
        "metadata": metadata,
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    print(f"[+] Experiment results saved in: {filepath}")
    return filepath

def setup_worker_env():
    """Prevents PyTorch/NumPy from crashing the CPU with too many internal threads."""
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def calculate_metrics(gt_img, res_img):
    """Returns RMSE and SSIM for a given estimation."""
    rmse = rmse_loss(gt_img, res_img)
    data_range = gt_img.max() - gt_img.min()
    
    # Prevent ssim from breaking if the map is completely flat
    data_range = data_range if data_range > 0 else 1.0
    
    ssim_val = ssim(gt_img, res_img, data_range=data_range)
    return rmse, ssim_val

def warmup_worker(cfg, seed):
    """Executes a tiny run to wake up JIT/PyTorch and prevent cold start penalties."""
    warmup_cfg = copy.deepcopy(cfg)
    warmup_res = 20
    warmup_cfg.env.cell_size = warmup_cfg.env.map_size[0] / warmup_res
    warmup_cfg.train.iterations = 5 
    
    warmup_batch, warmup_environment = generate_simulation_data(warmup_cfg)
    _ = create_system_matrix_sparse(
        (warmup_res, warmup_res), 
        warmup_batch.beams.tolist(),
        warmup_cfg.env.cell_size,
        quiet=True
    ).tocsr()

    run_gas_splatting(batch=warmup_batch, cfg=warmup_cfg, environment=warmup_environment)

def yield_parallel_experiment(worker_func, seeds, max_workers=4, **kwargs):
    """
    Agnostic parallel orchestrator that YIELDS results on the fly.
    The worker_func must accept 'seed' as its first argument, followed by **kwargs.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_func, seed, **kwargs): seed 
            for seed in seeds
        }
        
        for future in as_completed(futures):
            yield future.result()

def merge_local_results(results, local_results):
    """Merge per-seed local results into the shared experiment summary."""
    for metric, metric_data in local_results.items():
        for method, method_value in metric_data.items():
            if isinstance(method_value, dict):
                for x_value, value in method_value.items():
                    results[metric][method][x_value].append(value)
            else:
                results[metric][method].append(method_value)

# ==========================================
#            SCENARIO EXPERIMENTS
# ==========================================

def gas_map_metrics(gt: np.ndarray, pred: np.ndarray, obstacles: np.ndarray | None = None) -> dict:
    """Reconstruction metrics, computed on the free cells where gas can be.

    nrmse: RMSE divided by the ground-truth peak, comparable across scenarios.
    peak_ratio: reconstructed peak / true peak.
    mass_error: relative error of the total amount of gas.
    leak: fraction of the reconstructed gas that lies inside obstacles.
    """
    free = np.ones(gt.shape, dtype=bool) if obstacles is None else ~(obstacles > 0.5)
    g, p = gt[free], pred[free]
    peak = max(float(g.max()), 1e-12)
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    return {
        "rmse": rmse,
        "nrmse": rmse / peak,
        "peak_ratio": float(p.max()) / peak,
        "mass_error": float(p.sum() - g.sum()) / max(float(g.sum()), 1e-12),
        "leak": float(pred[~free].sum()) / max(float(pred.sum()), 1e-12),
    }


def run_scenario(cfg: Config) -> tuple[dict, np.ndarray]:
    """Simulates the measurements of cfg.sim.scenario, fits GASPLAT to them and evaluates it.

    Returns the metrics (see gas_map_metrics, plus model and run statistics)
    and the reconstructed map at the cell size of the scenario.
    """
    batch, environment = generate_simulation_data(cfg)

    t_start = time.time()
    model, _ = setup_gs_model(batch, cfg)
    trainer = Trainer(model, cfg, environment=environment)
    trainer.train(batch)
    training = trainer.finish()
    fit_time = time.time() - t_start

    pred = model.render_map(cell_size=cfg.env.cell_size)
    metrics = gas_map_metrics(environment.ground_truth.gas_map, pred, environment.obstacles)
    metrics.update({
        "scenario": os.path.basename(os.path.normpath(cfg.sim.scenario)) if cfg.sim.scenario else None,
        "beams": int(batch.measurements.shape[0]),
        "poses": int(batch.pose_ids.max()) + 1 if batch.pose_ids is not None else None,
        "gaussians": model.num_gaussians,
        "iterations": len(training.loss_history),
        "final_loss": training.loss_history[-1],
        "concentration_scale": model.concentration_scale,
        "time": fit_time,
    })
    return metrics, pred


def _run_scenario_task(task):
    key, cfg = task
    setup_worker_env()
    metrics, pred = run_scenario(cfg)
    return key, metrics, pred


def yield_scenario_runs(tasks: list[tuple], max_workers: int = 4):
    """Runs run_scenario for every (key, cfg) task in parallel processes and
    yields (key, metrics, pred) as they finish.

    The processes are spawned rather than forked, so that each one can use CUDA.
    With several workers the measured times are only indicative.
    """
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = [executor.submit(_run_scenario_task, (key, copy.deepcopy(cfg))) for key, cfg in tasks]
        for future in as_completed(futures):
            yield future.result()
