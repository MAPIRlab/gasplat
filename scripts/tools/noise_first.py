"""
Incremental run where the first batches carry no gas, only noise: the poses
are visited from the farthest to the closest to the source. The concentration
scale is first estimated from noise, far below the plume, and has to be raised
when the gas is found.

Three runs on the same measurements are compared:
  - incremental, raising the scale when a batch exceeds it,
  - incremental with the scale fixed by the first batch (no update),
  - every batch at once, as a reference.
Raising the scale should end close to the reference, while a fixed scale
leaves every relative threshold off by the ratio between noise and gas.

    python scripts/tools/noise_first.py [--scenario DIR] [--num_batches 5] [--noise_floor 2.0]

Without --scenario, the first scenario of the tuning set is used. Noise is
always added (absolute noise_floor, 2 ppm*m by default, plus the relative snr_db).
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config, EnvironmentParams, SimulationParams, TrainParams, DensificationParams
from trainer import Trainer
from utils.paths import PLOTS_DIR
from utils.metrics import gas_map_metrics
from utils.init_utils import setup_gs_model
from utils.scenario_sets import scenario_dirs
from utils.figure_style import apply_plot_style
from utils.experiment_utils import batch_config
from utils.sim_utils import MeasurementBatch, generate_simulation_data


def split_by_distance(batch: MeasurementBatch, source: np.ndarray, num_batches: int) -> list[MeasurementBatch]:
    """Groups the poses into batches, from the farthest to the closest to the source."""
    num_poses = int(batch.pose_ids.max()) + 1
    origins = np.array([batch.beams[batch.pose_ids == p][0, 0].tolist() for p in range(num_poses)])
    order = np.argsort(-np.linalg.norm(origins - source, axis=1))
    batches = []
    for group in np.array_split(order, num_batches):
        mask = torch.isin(batch.pose_ids, torch.as_tensor(group, device=batch.pose_ids.device))
        batches.append(MeasurementBatch(batch.beams[mask], batch.measurements[mask], batch.pose_ids[mask]))
    return batches


def run(batches: list[MeasurementBatch], environment, cfg: Config, fixed_scale: bool = False) -> list[dict]:
    """Trains on the batches in order and evaluates the model after each one.
    fixed_scale keeps the concentration scale set by the first batch."""
    model, trainer, history = None, None, []
    for b in batches:
        if model is None:
            model, _ = setup_gs_model(b, cfg)
            trainer = Trainer(model, cfg, environment=environment)
            if fixed_scale:
                trainer.update_concentration_scale = lambda batch: None
        trainer.train(b)
        pred = model.render_map(cfg.env.cell_size)
        metrics = gas_map_metrics(environment.ground_truth.gas_map, pred, environment.obstacles)
        metrics.update(scale=model.concentration_scale, gaussians=model.num_gaussians)
        history.append(metrics)
    return history


def main():
    parser = ArgumentParser(description="Incremental run whose first batches only measure noise")
    # Noise always on, with an absolute floor by default (see above)
    default = Config(EnvironmentParams(), SimulationParams(noise=True, noise_floor=2.0), TrainParams(),
                     DensificationParams())
    parser.add_arguments(Config, dest="cfg", default=default)
    parser.add_argument("--num_batches", type=int, default=5, help="Batches of poses, visited one after the other")
    args = parser.parse_args()
    cfg: Config = batch_config(args.cfg)
    if not cfg.sim.scenario:
        cfg.sim.scenario = scenario_dirs("tuning")[0]
    apply_plot_style()

    batch, environment = generate_simulation_data(cfg)
    with open(os.path.join(cfg.sim.scenario, "meta.json")) as f:
        source = np.array(json.load(f)["source_local"])
    batches = split_by_distance(batch, source, args.num_batches)

    runs = {
        "incremental, scale raised": run(batches, environment, cfg),
        "incremental, scale fixed by batch 1": run(batches, environment, cfg, fixed_scale=True),
    }
    reference = run([batch], environment, cfg)[0]

    name = os.path.basename(os.path.normpath(cfg.sim.scenario))
    print(f"{name}: {args.num_batches} batches, noise floor {cfg.sim.noise_floor} ppm*m")
    for label, history in runs.items():
        print(f"\n{label}")
        for i, h in enumerate(history):
            print(f"  batch {i + 1}: scale {h['scale']:9.3f} ppm  gaussians {h['gaussians']:4d}  "
                  f"NRMSE {100 * h['nrmse']:5.2f}%  peak x{h['peak_ratio']:.2f}")
    print(f"\nall at once: scale {reference['scale']:.3f} ppm  gaussians {reference['gaussians']}  "
          f"NRMSE {100 * reference['nrmse']:.2f}%  peak x{reference['peak_ratio']:.2f}")

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(1, args.num_batches + 1)
    for (key, label, factor), ax in zip([("nrmse", "NRMSE (% of peak)", 100), ("gaussians", "Gaussians", 1),
                                         ("scale", "Concentration scale (ppm)", 1)], axs):
        for run_label, history in runs.items():
            ax.plot(x, [h[key] * factor for h in history], marker='o', label=run_label)
        ax.axhline(reference[key] * factor, color='k', ls='--', label="all at once")
        ax.set_xlabel("Batch (poses farthest from the source first)")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.grid(True, ls="--", alpha=0.3)
    axs[2].set_yscale('log')
    axs[0].legend(fontsize=10)
    fig.suptitle(f"Noise-first incremental run on {name.split('_h')[0]}")
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'noise_first.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[+] Plot saved in: {save_path}")


if __name__ == "__main__":
    main()
