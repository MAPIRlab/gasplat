import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

from config import Config
from gs_model import GasSplattingModel
from trainer import TrainingResults
from utils.sim_utils import EnvironmentContext, MeasurementBatch
from utils.init_utils import InitializationData

def set_publication_style():
    """Configures Matplotlib global style"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'grid.alpha': 0.7,
        'grid.linestyle': '--',
        # --- Saving configuration ---
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'savefig.pad_inches': 0.05
    })

def plot_initial_guess(img_gt: np.ndarray | None, init_data: InitializationData, cfg: Config):
    """Shows ground truth (if available) and initial reconstruction seeds, and saves it."""

    map_w, map_h = cfg.env.map_size

    init_pos = init_data.pos.detach().cpu().numpy()
    
    # Create a blank dark background to frame the seeds nicely
    grid_w, grid_h = round(map_w / cfg.env.cell_size), round(map_h / cfg.env.cell_size)
    background = np.zeros((grid_h, grid_w))

    if img_gt is not None:
        # Create a 2-panel side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        vmax = max(img_gt.max(), 1e-3) # Prevent vmax=0 warnings

        ax1.set_title(f"Ground Truth ({img_gt.shape[0]}x{img_gt.shape[1]})")
        im = ax1.imshow(img_gt, origin='lower', extent=(0, map_w, 0, map_h), cmap='viridis', vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax1, label="ppm", fraction=0.046, pad=0.04)

        ax2.set_title("Initialization")
        ax2.imshow(background, origin='lower', extent=(0, map_w, 0, map_h), cmap='viridis', vmin=0, vmax=1)
        ax2.scatter(init_pos[:, 0], init_pos[:, 1], marker='X', c='w', edgecolors='k', s=90, linewidths=1.2, label='Positions')
        ax2.legend()

    else:
        # Create a single-panel plot just for the initialization
        fig, ax2 = plt.subplots(1, 1, figsize=(7, 5))
        
        ax2.set_title("Initialization")
        ax2.imshow(background, origin='lower', extent=(0, map_w, 0, map_h), cmap='viridis', vmin=0, vmax=1)
        ax2.scatter(init_pos[:, 0], init_pos[:, 1], marker='X', c='w', edgecolors='k', s=90, linewidths=1.2, label='Positions')
        ax2.legend()
        
    # Save plot instead of showing it
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'initial_guess.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    print(f"[+] Initial guess plot saved in: {save_path}")
    
    plt.close(fig) # Free memory

def draw_beams(ax, batch: MeasurementBatch, by_measurement: bool = False):
    """Draws the beams and the vantage poses (mean sensor position of each pose).

    Lines get fainter as beams get denser (1/sqrt(N)), so that a fine scan does
    not hide the map. by_measurement: opacity grows with the measured integral
    instead, which shows where the gas is when there is no ground truth.
    """
    beams = batch.beams.cpu().numpy()
    if by_measurement:
        meas = batch.measurements.cpu().numpy()
        span = meas.max() - meas.min() if len(meas) else 0.0
        meas_norm = (meas - meas.min()) / span if span > 0 else np.zeros_like(meas)
        colors = np.zeros((len(beams), 4))
        colors[:, :3] = 1.0
        colors[:, 3] = 0.01 + meas_norm * (0.05 - 0.01)
        ax.add_collection(LineCollection(beams, colors=colors, linewidths=0.5))
    else:
        alpha = float(np.clip(4.0 / np.sqrt(max(len(beams), 1)), 0.08, 0.5))
        ax.add_collection(LineCollection(beams, colors='w', linewidths=0.5, alpha=alpha))
    if batch.pose_ids is not None:
        pose_ids = batch.pose_ids.cpu().numpy()
        poses = np.array([beams[pose_ids == p, 0].mean(axis=0) for p in np.unique(pose_ids)])
        ax.scatter(poses[:, 0], poses[:, 1], marker='^', s=18, c='w', edgecolors='k', linewidths=0.5, zorder=3)

def plot_training_results(gaussians: GasSplattingModel, batch: MeasurementBatch, environment: EnvironmentContext, results: TrainingResults, cfg: Config):
    """Saves a plot with GT (if available), GS reconstruction, loss history (with optional RMSE), and densification events.

    Without a ground truth (real measurements), the beams are drawn over the reconstruction instead.
    """

    map_w, map_h = cfg.env.map_size
    grid_w = round(map_w / cfg.env.cell_size)
    grid_h = round(map_h / cfg.env.cell_size)
    max_map_dim = max(map_w, map_h)

    # Generate images
    img_pred_gaussian = gaussians.render_map(cell_size=max_map_dim / 100)
    img_pred = gaussians.render_map(cell_size=cfg.env.cell_size)

    # Colormap min and max values
    vmin = 0
    ground_truth = environment.ground_truth.gas_map if environment.ground_truth is not None else None
    vmax = img_pred.max() if ground_truth is None else max(ground_truth.max(), img_pred.max())

    obstacles = environment.obstacles
    walls = None if obstacles is None else np.ma.masked_where(~(obstacles > 0.5), np.ones(obstacles.shape))

    fig = plt.figure(figsize=(12, 8))

    # Grid of 3 rows. Images on top, Loss in the middle, Densification below
    num_maps = 3 if ground_truth is not None else 2
    gs = gridspec.GridSpec(3, num_maps, height_ratios=[1.5, 1, 1], hspace=0.3)
    map_axes = []

    # 1. GT (Top Left)
    if ground_truth is not None:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(f"Ground Truth ({grid_w}x{grid_h})")
        ax1.imshow(ground_truth, origin='lower', extent=(0, map_w, 0, map_h), cmap='viridis', vmin=vmin, vmax=vmax)
        draw_beams(ax1, batch)
        map_axes.append(ax1)

    # 2. Reconstruction (Top Center)
    ax2 = fig.add_subplot(gs[0, num_maps - 2])
    pos = gaussians.get_pos().detach().cpu().numpy()
    ax2.set_title("GS Reconstruction")
    ax2.imshow(img_pred_gaussian, origin='lower', extent=(0, map_w, 0, map_h), cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
    if ground_truth is None:
        draw_beams(ax2, batch, by_measurement=True)
    ax2.scatter(pos[:, 0], pos[:, 1], marker='P', c='k', edgecolors='w', s=30, linewidths=1.0)
    map_axes.append(ax2)

    # 3. Reconstruction Grid (Top Right), with the RMSE over free cells only when the environment has obstacles
    ax3 = fig.add_subplot(gs[0, num_maps - 1])
    if ground_truth is not None:
        free = np.ones_like(ground_truth, dtype=bool) if obstacles is None else ~(obstacles > 0.5)
        rmse = np.sqrt(np.mean((img_pred - ground_truth)[free]**2))
        nrmse = rmse / max(ground_truth[free].max(), 1e-12)
        ax3.set_title(f"GS Reconstruction (Grid)\nRMSE = {rmse:.4g} ({100 * nrmse:.1f}% of peak)")
    else:
        ax3.set_title(f"GS Reconstruction (Grid, {grid_w}x{grid_h})")
    im3 = ax3.imshow(img_pred, origin='lower', extent=(0, map_w, 0, map_h), cmap='viridis', vmin=vmin, vmax=vmax)
    map_axes.append(ax3)

    if walls is not None:
        for ax in map_axes:
            ax.imshow(walls, origin='lower', extent=(0, map_w, 0, map_h), cmap='Greys', vmin=0, vmax=2.5, interpolation='nearest')

    # Single Colorbar
    fig.colorbar(im3, ax=map_axes, label="ppm", fraction=0.015, pad=0.02)

    # ==========================================
    # 4. Loss History (Middle Row)
    # ==========================================
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title(f"Loss History (Final Loss = {results.loss_history[-1]:.4f})")
    
    # Plot standard Loss on the left axis
    line_loss = ax4.plot(results.loss_history, color='blue', alpha=0.6, label="Total Loss (relative L1)")
    ax4.set_ylabel("Total Loss", color='blue')
    ax4.set_yscale('log')
    ax4.set_xlim(0, len(results.loss_history))
    ax4.grid(True, which="both", ls="--", alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.tick_params(labelbottom=False)

    lines = line_loss
    
    # If rmse_history has data, plot it on a twin right axis
    if results.rmse_history:
        ax4_rmse = ax4.twinx()
        iters = list(results.rmse_history.keys())
        rmse_vals = list(results.rmse_history.values())
        
        line_rmse = ax4_rmse.plot(iters, rmse_vals, color='red', alpha=0.8, linewidth=2, marker='o', markersize=4, label="Spatial RMSE")
        ax4_rmse.set_ylabel("RMSE (ppm)", color='red')
        ax4_rmse.tick_params(axis='y', labelcolor='red')
        
        lines += line_rmse # Combine lines for a single legend
        
    labels = [str(l.get_label()) for l in lines]
    ax4.legend(lines, labels, loc='upper right')

    # ==========================================
    # 5. Densification Events (Bottom Row)
    # ==========================================
    ax5 = fig.add_subplot(gs[2, :], sharex=ax4)
    ax5.set_title("Densification Events")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Count")

    if results.densify_history:
        iters = list(results.densify_history.keys())
        clones = [d.get('clones', 0) for d in results.densify_history.values()]
        splits = [d.get('splits', 0) for d in results.densify_history.values()]
        prunes = [d.get('prunes', 0) for d in results.densify_history.values()]
        injections = [d.get('injections', 0) for d in results.densify_history.values()]

        bar_width = cfg.densify.densify_interval * 0.4 
        
        # Track the bottom of the stack to build a proper stacked bar chart
        bottom_stack = np.zeros(len(iters))

        # Dynamic plotting: Only plot and add to legend if the event actually occurred
        has_events = False
        if sum(clones) > 0:
            ax5.bar(iters, clones, width=bar_width, bottom=bottom_stack, label='Clones', color='skyblue')
            bottom_stack += np.array(clones)
            has_events = True
            
        if sum(splits) > 0:
            label = 'Splits' if cfg.densify.original_dens else 'Splits (Long-Axis)'
            ax5.bar(iters, splits, width=bar_width, bottom=bottom_stack, label=label, color='orange')
            bottom_stack += np.array(splits)
            has_events = True
            
        if sum(injections) > 0:
            ax5.bar(iters, injections, width=bar_width, bottom=bottom_stack, label='Injections', color='purple')
            bottom_stack += np.array(injections)
            has_events = True
            
        if sum(prunes) > 0:
            ax5.bar(iters, prunes, width=bar_width, bottom=bottom_stack, label='Prunes', color='red')
            has_events = True

        if has_events: ax5.legend(loc='upper right')
        
    ax5.grid(True, axis='y', ls="--", alpha=0.3)

    # Save plot instead of showing it
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'training_results.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[+] Training results plot saved in: {save_path}")
    
    plt.close(fig) # Free memory

def plot_reconstruction_grid(cases: list[str], maps: dict, rows: dict, titles: dict, save_path: str, title: str = ""):
    """Ground truth (top row) and one row per method, one column per case.

    maps[case] holds "gt", "obstacles" (or None) and "extent"; rows maps a method
    name to {case: reconstruction}; titles maps (method, case) to a subplot title.
    Every column shares the color scale of its ground truth.
    """
    n_rows, n_cols = len(rows) + 1, len(cases)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.0 * n_rows + 0.5), squeeze=False)
    for j, case in enumerate(cases):
        m = maps[case]
        free = np.ones(m["gt"].shape, dtype=bool) if m["obstacles"] is None else ~(m["obstacles"] > 0.5)
        vmax = m["gt"][free].max()
        walls = np.ma.masked_where(free, np.ones_like(m["gt"]))
        panels = [("Ground truth", m["gt"], case.split("_h")[0])] + \
                 [(method, preds[case], titles.get((method, case), "")) for method, preds in rows.items()]
        for i, (row_label, img, label) in enumerate(panels):
            ax = axs[i, j]
            ax.imshow(np.ma.masked_where(~free, img), origin='lower', extent=m["extent"], cmap='viridis', vmin=0, vmax=vmax)
            ax.imshow(walls, origin='lower', extent=m["extent"], cmap='Greys', vmin=0, vmax=2.5, interpolation='nearest')
            ax.set_title(label, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_label)
    if title:
        fig.suptitle(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[+] Reconstruction grid saved in: {save_path}")


def plot_sweep(param: str, values: list, runs: dict, default, save_path: str, baseline: dict | None = None):
    """Metrics of a hyperparameter sweep: one thin line per scenario, the mean in bold.

    runs maps (value, scenario) -> metrics dict. The current default value is
    marked with a dotted line. When it is not a number (e.g. None), the mean of
    the baseline runs (scenario -> metrics) is drawn as a dashed line instead.
    """
    scenarios = sorted({s for _, s in runs})
    panels = [("nrmse", "NRMSE (% of peak)", 100), ("peak_ratio", "Peak ratio", 1),
              ("mass_error", "Mass error (%)", 100), ("leak", "Gas inside obstacles (%)", 100),
              ("gaussians", "Gaussians", 1), ("time", "Fit time (s)", 1)]
    log_x = min(values) > 0 and max(values) / min(values) >= 8
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (key, label, factor) in zip(axs.ravel(), panels):
        table = np.array([[runs[(v, s)][key] * factor for v in values] for s in scenarios])
        for i, s in enumerate(scenarios):
            ax.plot(values, table[i], color=colors[i % 10], lw=1, alpha=0.6, marker='.', label=s.split("_h")[0])
        ax.plot(values, table.mean(axis=0), color='k', lw=2.5, marker='o', label="mean")
        if key == "peak_ratio":
            ax.axhline(1.0, color='gray', lw=1)
        if baseline is not None:
            ax.axhline(np.mean([baseline[s][key] * factor for s in scenarios]), color='k', ls='--', lw=1.5,
                       label=f"current ({default})")
        elif default is not None:
            ax.axvline(default, color='gray', ls=':', lw=1.5)
        if log_x:
            ax.set_xscale('log')
        ax.set_xlabel(param)
        ax.set_ylabel(label)
        ax.grid(True, which="both", ls="--", alpha=0.3)
    axs[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Sweep of {param} (current value: {default})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[+] Sweep plot saved in: {save_path}")
