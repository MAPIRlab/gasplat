import os
import copy
import json
import time
import datetime
import multiprocessing
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.scenario_sets import case_name
from utils.sim_utils import EnvironmentContext, MeasurementBatch, generate_simulation_data
from methods.tomography import create_system_matrix_sparse
from methods.registry import AVAILABLE_METHODS

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

# Thread pools of the numerical libraries, read once when they are loaded
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def setup_worker_env(threads: int = 1):
    """Limits the PyTorch threads of a worker process. NumPy/SciPy read
    THREAD_VARIABLES when they are imported, so yield_runs sets those before
    starting the workers."""
    torch.set_num_threads(threads)

# ==========================================
#            SCENARIO EXPERIMENTS
# ==========================================

def gas_map_metrics(gt: np.ndarray, pred: np.ndarray, obstacles: np.ndarray | None = None,
                    cell_size: float = 1.0) -> dict:
    """Reconstruction metrics, computed on the free cells where gas can be.

    nrmse: RMSE divided by the ground-truth peak, comparable across scenarios.
    peak_ratio: reconstructed peak / true peak.
    peak_distance: distance between the true and the reconstructed maxima, in the
        units of cell_size (meters for scenarios).
    mass_error: relative error of the total amount of gas.
    leak: fraction of the reconstructed gas that lies inside obstacles.
    """
    free = np.ones(gt.shape, dtype=bool) if obstacles is None else ~(obstacles > 0.5)
    g, p = gt[free], pred[free]
    peak = max(float(g.max()), 1e-12)
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    true_max = np.unravel_index(np.argmax(np.where(free, gt, -np.inf)), gt.shape)
    pred_max = np.unravel_index(np.argmax(np.where(free, pred, -np.inf)), gt.shape)
    return {
        "rmse": rmse,
        "nrmse": rmse / peak,
        "peak_ratio": float(p.max()) / peak,
        "peak_distance": float(np.hypot(true_max[0] - pred_max[0], true_max[1] - pred_max[1])) * cell_size,
        "mass_error": float(p.sum() - g.sum()) / max(float(g.sum()), 1e-12),
        "leak": float(pred[~free].sum()) / max(float(pred.sum()), 1e-12),
    }


def evaluation_maps(environment: EnvironmentContext, cfg: Config) -> dict:
    """Ground truth, obstacles and extent of a case, to plot reconstructions next to it."""
    return {"gt": environment.ground_truth.gas_map, "obstacles": environment.obstacles,
            "extent": (0, cfg.env.map_size[0], 0, cfg.env.map_size[1])}


def run_scenario(cfg: Config) -> tuple[dict, dict]:
    """Simulates the measurements of a case, fits GASPLAT to them and evaluates it.

    Returns the metrics (see gas_map_metrics, plus model and run statistics)
    and the maps (evaluation_maps plus the reconstruction "pred").
    """
    batch, environment = generate_simulation_data(cfg)

    model, init_data = setup_gs_model(batch, cfg)
    t0 = time.time()
    trainer = Trainer(model, cfg, environment=environment)   # precomputes the distance fields
    trainer_setup_time = time.time() - t0
    trainer.train(batch)
    training = trainer.finish()
    # Setup and optimization time, as measured by the initialization and the trainer
    fit_time = init_data.info["setup_time"] + trainer_setup_time + training.training_time

    pred = model.render_map(cell_size=cfg.env.cell_size)
    metrics = gas_map_metrics(environment.ground_truth.gas_map, pred, environment.obstacles, cfg.env.cell_size)
    metrics.update({
        "scenario": case_name(cfg),
        "beams": int(batch.measurements.shape[0]),
        "poses": int(batch.pose_ids.max()) + 1 if batch.pose_ids is not None else None,
        "gaussians": model.num_gaussians,
        "iterations": len(training.loss_history),
        "final_loss": training.loss_history[-1],
        "concentration_scale": model.concentration_scale,
        "time": fit_time,
    })
    return metrics, {**evaluation_maps(environment, cfg), "pred": pred}


def measurements_at(cfg: Config, step: float, axis: str) -> tuple[MeasurementBatch, EnvironmentContext]:
    """Measurements of one point of the experiment on the number of measurements
    (scripts/experiments/num_measurements.py): `step` poses or a sweep every `step`
    degrees on a GADEN scenario (axis "poses" or "angular_step"), or the first
    `step` beams of a fixed random order of cfg.sim.num_beams on a procedural map
    (axis "beams"). Modifies cfg."""
    if axis == "poses":
        cfg.sim.num_poses = int(step)
        return generate_simulation_data(cfg)
    if axis == "angular_step":
        cfg.sim.angular_step = step
        return generate_simulation_data(cfg)
    cfg.sim.num_beams = max(cfg.sim.num_beams, int(step))
    batch, environment = generate_simulation_data(cfg)
    order = torch.as_tensor(np.random.default_rng(cfg.sim.seed).permutation(len(batch.measurements))[:int(step)],
                            device=batch.beams.device)
    return MeasurementBatch(batch.beams[order], batch.measurements[order]), environment


def run_methods(batch: MeasurementBatch, environment: EnvironmentContext, cfg: Config,
                methods: list[str]) -> tuple[dict, dict]:
    """Reconstructs the same measurements with several methods (see methods/registry.py).

    The system matrix of the grid methods is built once; its time is reported
    on its own ("matrix_time"), and each method reports only its own solve or
    fit time. Returns ({method: metrics}, {method: reconstruction}).
    """
    gt = environment.ground_truth.gas_map
    system_matrix, matrix_time = None, None
    if any(AVAILABLE_METHODS[m].get("needs_matrix") for m in methods):
        t_start = time.time()
        system_matrix = create_system_matrix_sparse(gt.shape, batch.beams.cpu().numpy(), cfg.env.cell_size)
        matrix_time = time.time() - t_start

    metrics, preds = {}, {}
    for name in methods:
        pred, solve_time = AVAILABLE_METHODS[name]["func"](batch=batch, cfg=cfg, environment=environment,
                                                           system_matrix=system_matrix)
        metrics[name] = {**gas_map_metrics(gt, pred, environment.obstacles, cfg.env.cell_size), "time": solve_time,
                         "matrix_time": matrix_time if AVAILABLE_METHODS[name].get("needs_matrix") else None}
        preds[name] = pred
    return metrics, preds


# ==========================================
#              PARALLEL RUNNER
# ==========================================

_warmed_up = False


def _warm_up(cfg: Config):
    """One short throwaway run of every method on the device of cfg, once per
    process, so that CUDA initialization does not count in the first timed run."""
    global _warmed_up
    if _warmed_up:
        return
    warm_cfg = copy.deepcopy(cfg)
    warm_cfg.sim.scenario, warm_cfg.sim.seed, warm_cfg.sim.num_beams = None, 0, 30
    warm_cfg.env.map_size, warm_cfg.env.cell_size = (10.0, 10.0), 1.0
    warm_cfg.train.iterations, warm_cfg.quiet = 20, True
    batch, environment = generate_simulation_data(warm_cfg)
    run_methods(batch, environment, warm_cfg, list(AVAILABLE_METHODS))
    _warmed_up = True


def _run_task(task):
    func, key, cfg, kwargs, warmup, threads = task
    setup_worker_env(threads)
    if warmup:
        _warm_up(cfg)
    try:
        result = func(cfg, **kwargs)
    except torch.OutOfMemoryError:
        # Runs that grow too many Gaussians for the GPU memory finish on the CPU
        torch.cuda.empty_cache()
        cfg.device_type = "cpu"
        result = func(cfg, **kwargs)
    return key, result, cfg.device_type


def default_workers() -> int:
    """One worker per GPU (runs sharing a GPU only take longer each), or 4 on the CPU."""
    return torch.cuda.device_count() or 4


def yield_runs(func, tasks: list[tuple], max_workers: int = 4, warmup: bool = False):
    """Runs func(cfg, **kwargs) for every (key, cfg) or (key, cfg, kwargs) task in
    parallel processes and yields (key, result, device) as they finish.

    func must be defined at module level. The processes are spawned rather than
    forked, so that each one can use CUDA, and the runs are spread over all the
    available GPUs. A run that does not fit in GPU memory is repeated on the
    CPU. The CPU cores are shared out among the workers, so that their thread
    pools do not oversubscribe the machine. With warmup, every process makes a
    throwaway run before its first timed one.

    Times: a single worker keeps every run on the same GPU, so that times are
    comparable; with one worker per GPU no run waits for another; with more
    workers than GPUs, runs share a GPU and their times grow accordingly.
    """
    num_gpus = torch.cuda.device_count()
    threads = max(1, (os.cpu_count() or 1) // max_workers)
    saved = {v: os.environ.get(v) for v in THREAD_VARIABLES}
    os.environ.update({v: str(threads) for v in THREAD_VARIABLES})  # inherited by the spawned workers
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            futures = []
            for i, task in enumerate(tasks):
                key, cfg, kwargs = task if len(task) == 3 else (*task, {})
                cfg = copy.deepcopy(cfg)
                if cfg.device_type == "cuda" and num_gpus > 1 and max_workers > 1:
                    cfg.device_type = f"cuda:{i % num_gpus}"
                futures.append(executor.submit(_run_task, (func, key, cfg, kwargs, warmup, threads)))
            for future in as_completed(futures):
                yield future.result()
    finally:
        for v, value in saved.items():
            if value is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = value


def yield_scenario_runs(tasks: list[tuple], max_workers: int = 4, warmup: bool = True):
    """yield_runs of run_scenario: yields (key, metrics, maps), with the device in the metrics."""
    for key, (metrics, maps), device in yield_runs(run_scenario, tasks, max_workers, warmup):
        metrics["device"] = device
        yield key, metrics, maps
