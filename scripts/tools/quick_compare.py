"""
Runs every method (or --methods) on the measurements of one case and plots
their reconstructions side by side, to judge them at a glance. Works on any
case, including real data, which has no ground truth: the first panel then
shows the beams, more opaque the higher their reading, and every map gets its
own color scale (its peak is in the title), so that a method with a few sharp
spikes does not hide the others. With a ground truth, it comes first, every
map shares its color scale, and the NRMSE of each method is shown.

    python scripts/tools/quick_compare.py --scenario real_data/full_sweep
    python scripts/tools/quick_compare.py --scenario data/scenarios/house01_2,4-1_slow_h0.85_it1999 --num_poses 6
    python scripts/tools/quick_compare.py --seed 3 --methods LFD "Gas Splatting"     # procedural map
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from methods.registry import AVAILABLE_METHODS
from utils.scenario_sets import case_name
from utils.sim_utils import generate_simulation_data
from utils.plot_utils import draw_beams, set_publication_style
from utils.experiment_utils import RUN_THREADS, run_methods


def main():
    parser = ArgumentParser(description="Reconstructions of every method on one case, side by side")
    parser.add_arguments(Config, dest="cfg")
    parser.add_argument("--methods", nargs="+", default=list(AVAILABLE_METHODS), choices=list(AVAILABLE_METHODS))
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    torch.set_num_threads(RUN_THREADS)
    set_publication_style()

    batch, environment = generate_simulation_data(cfg)
    metrics, preds = run_methods(batch, environment, cfg, args.methods)
    name = case_name(cfg)
    gt = environment.ground_truth.gas_map if environment.ground_truth is not None else None
    extent = (0, cfg.env.map_size[0], 0, cfg.env.map_size[1])
    shape = next(iter(preds.values())).shape
    free = np.ones(shape, dtype=bool) if environment.obstacles is None else ~(environment.obstacles > 0.5)
    walls = np.ma.masked_where(free, np.ones(shape))

    print(f"{name}: {len(batch.measurements)} beams")
    print(f"  {'method':18s} {'NRMSE %':>8s} {'time s':>8s}")
    for m in args.methods:
        x = metrics[m]
        nrmse = f"{100 * x['nrmse']:8.2f}" if gt is not None else f"{'-':>8s}"
        print(f"  {m:18s} {nrmse} {x['time'] + (x['matrix_time'] or 0.0):8.2f}")

    # First panel: the ground truth, or the beams over an empty map
    first = ("Ground truth", gt if gt is not None else np.zeros(shape))
    panels = [first] + [(m, preds[m]) for m in args.methods]
    cols = min(3, len(panels))
    rows = -(-len(panels) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, (4.4 * shape[0] / shape[1] + 0.9) * rows), squeeze=False)
    for ax in axs.flat[len(panels):]:
        ax.axis("off")
    for ax, (title, img) in zip(axs.flat, panels):
        vmax = gt[free].max() if gt is not None else max(float(img[free].max()), 1e-12)
        im = ax.imshow(np.ma.masked_where(~free, img), origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=vmax)
        ax.imshow(walls, origin='lower', extent=extent, cmap='Greys', vmin=0, vmax=2.5, interpolation='nearest')
        if title in metrics:
            x = metrics[title]
            time_s = x["time"] + (x["matrix_time"] or 0.0)
            title += f"\nNRMSE {100 * x['nrmse']:.1f}%, {time_s:.2f} s" if gt is not None else \
                     f"\npeak {vmax:.3g} ppm, {time_s:.2f} s"
            if gt is None:   # own scale: a colorbar of the map's height
                fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05))
        elif gt is None:
            title = f"Beams ({len(batch.measurements)})"
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    draw_beams(axs[0, 0], batch, by_measurement=gt is None)
    if gt is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(), label="ppm", fraction=0.02, pad=0.02)
    fig.suptitle(name)

    save_path = os.path.join(os.path.dirname(__file__), '..', '..', 'plots', f'quick_compare_{name}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[+] Reconstructions saved in: {save_path}")


if __name__ == "__main__":
    main()
