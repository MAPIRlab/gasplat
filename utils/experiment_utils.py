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
from trainer import fit_gasplat
from utils.paths import RESULTS_DIR
from utils.metrics import gas_map_metrics
from utils.scenario_sets import case_name
from utils.sim_utils import EnvironmentContext, MeasurementBatch, generate_simulation_data
from methods.tomography import create_system_matrix_sparse
from methods.registry import AVAILABLE_METHODS

def save_experiment_results(metadata, results, folder=RESULTS_DIR):
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


def batch_config(cfg: Config) -> Config:
    """Sets up cfg for unattended runs: no console output, no evaluation during training
    and no live plots. Returns cfg."""
    cfg.quiet, cfg.train.do_eval, cfg.train.live_vis = True, False, False
    return cfg


def total_time(metrics: dict) -> float:
    """Time of a method's reconstruction (run_methods): its own time plus the system
    matrix of the grid methods, built once for all of them."""
    return metrics["time"] + (metrics.get("matrix_time") or 0.0)

# Thread pools of the numerical libraries, read once when they are loaded
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# CPU threads of a run. More slow GASPLAT down (many small operations: 3.6 s per tuning
# house with 2-4 threads, 12.8 s with 16) and LFD/LTD too
RUN_THREADS = 4


def setup_worker_env(threads: int = 1):
    """Limits the PyTorch threads of a worker process. NumPy/SciPy read
    THREAD_VARIABLES when they are imported, so yield_runs sets those before
    starting the workers."""
    torch.set_num_threads(threads)

# ==========================================
#            SCENARIO EXPERIMENTS
# ==========================================

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

    model, training, fit_time = fit_gasplat(batch, cfg, environment)

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
                methods: list[str], levels: np.ndarray | None = None) -> tuple[dict, dict]:
    """Reconstructs the same measurements with several methods (see methods/registry.py).

    The system matrix of the grid methods is built once; its time is reported
    on its own ("matrix_time"), and each method reports only its own solve or
    fit time. Without a ground truth (real data) only the times are reported.
    With the observation levels of the cells, the metrics also cover each region
    (gas_map_metrics). Returns ({method: metrics}, {method: reconstruction}).
    """
    gt = environment.ground_truth.gas_map if environment.ground_truth is not None else None
    system_matrix, matrix_time = None, None
    if any(AVAILABLE_METHODS[m].get("needs_matrix") for m in methods):
        grid_shape = (round(cfg.env.map_size[1] / cfg.env.cell_size), round(cfg.env.map_size[0] / cfg.env.cell_size))
        t_start = time.time()
        system_matrix = create_system_matrix_sparse(grid_shape, batch.beams.cpu().numpy(), cfg.env.cell_size)
        matrix_time = time.time() - t_start

    metrics, preds = {}, {}
    for name in methods:
        pred, solve_time = AVAILABLE_METHODS[name]["func"](batch=batch, cfg=cfg, environment=environment,
                                                           system_matrix=system_matrix)
        map_metrics = (gas_map_metrics(gt, pred, environment.obstacles, cfg.env.cell_size, levels)
                       if gt is not None else {})
        metrics[name] = {**map_metrics, "time": solve_time,
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


def default_workers(cfg: Config) -> int:
    """On the GPU, one worker per GPU (runs sharing a GPU only take longer each); on the
    CPU, as many as fit with RUN_THREADS threads each."""
    if cfg.device.type == "cuda":
        return max(1, torch.cuda.device_count())
    return max(1, (os.cpu_count() or 1) // RUN_THREADS)


def yield_runs(func, tasks: list[tuple], max_workers: int = 4, warmup: bool = False):
    """Runs func(cfg, **kwargs) for every (key, cfg) or (key, cfg, kwargs) task in
    parallel processes and yields (key, result, device) as they finish.

    func must be defined at module level. The processes are spawned rather than
    forked, so that each one can use CUDA, and the runs are spread over all the
    available GPUs. A run that does not fit in GPU memory is repeated on the
    CPU. Every worker gets RUN_THREADS CPU threads, fewer if the cores do not
    go round, so that their thread pools do not oversubscribe the machine. With
    warmup, every process makes a throwaway run before its first timed one.

    Times: with a single worker, runs do not compete for cores and every run on
    the GPU uses the same one, so that times are comparable; with one worker per
    GPU no run waits for another; with more workers than GPUs, runs share a GPU
    and their times grow accordingly.
    """
    num_gpus = torch.cuda.device_count()
    threads = max(1, min(RUN_THREADS, (os.cpu_count() or 1) // max_workers))
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
