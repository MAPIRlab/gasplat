<h1 align="center">Gas Splatting</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white" alt="PyTorch">
</p>

<p align="center">
  <img src="assets/training_evolution.gif" alt="Gas Splatting Training Example" width="800">
</p>
<p align="center"><i>Example of the Gas Splatting training evolution.</i></p>

## 📖 Overview

This repository contains the code for my Bachelor's Thesis (Trabajo de Fin de Grado - TFG). It presents a novel approach to gas distribution mapping by introducing **Gas Splatting**, a method based on Gaussian Splatting for 2D gas tomography using Tunable Diode Laser Absorption Spectroscopy (TDLAS) measurements.

The primary goal of this project is to accurately and efficiently reconstruct gas concentration maps by optimizing a set of 2D Gaussians through a differentiable pipeline.

## 📊 Performance Comparison

To evaluate the effectiveness of **Gas Splatting**, it is compared against traditional and state-of-the-art tomographic methods. The following results were obtained on a simulated 20x20m map with a grid resolution of 0.5m, using 40 TDLAS beams.

<p align="center">
  <img src="assets/compare_methods.png" alt="Methods Comparison" width="600">
</p>

| Method | RMSE ↓ | SSIM ↑ | Total Time (s) |
| :--- | :---: | :---: | :---: |
| SART | 0.1381 | 0.1481 | 1.19 |
| RBF Coupled SART | 0.0685 | 0.4623 | 1.30 |
| LFD | 0.0736 | 0.4788 | 1.24 |
| LTD | 0.0723 | 0.4760 | 1.23 |
| **Gas Splatting** | **0.0311** | **0.9045** | 3.51 |

*Note: While Gas Splatting requires more computational time, it achieves a significantly higher reconstruction quality (SSIM) and lower error (RMSE) compared to baseline methods.*

## 🗂️ Repository Structure

```bash
gas_splatting/
├── methods/            # Reconstruction method registry and runners
├── real_data/          # Real-world TDLAS sweep datasets (JSON)
├── scripts/            # experiments/ (paper comparisons), figures/ (method figures), tools/ (tuning and checks)
├── utils/              # Simulation, geometry, initialization, and math utilities
├── visualization/      # Visualization components and interactive tools
├── config.py           # simple_parsing configuration definitions
├── gs_model.py         # Gas Splatting model, densification logic 
├── main.py             # Main entry point (Sim -> Init -> Train)
└── trainer.py          # Optimization loop
```

## 🛠️ Installation

**1. Clone repository**
```bash
git clone https://github.com/aleevr04/gas_splatting.git
cd gas_splatting
```

**2. Create and activate a virtual environment**

The most convenient way to install dependencies is by creating a virtual environment. There are several tools for this purpose like `venv` or `conda`. A `venv` example is shown below:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

**3. Install PyTorch**

Since PyTorch is highly hardware-dependent (CPU or GPU), please install it by following the instructions from their [official website](https://pytorch.org/get-started/locally/). CPU-only example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**4. Install project dependencies**

```bash
pip install -r requirements.txt
```

<details>
<summary><h2>🚀 Usage</h2></summary>

### Standard Pipeline

To run the default simulation and training pipeline, simply execute the main script:

```bash
python main.py
```

This will generate simulation data (ground truth, beam geometry, and measurements), initialize model parameters, and start training.

### Experiments, figures and tools

Every script runs on a set of cases: GADEN scenarios (`--set tuning`, `held_out`, `all`, listed in `utils/scenario_sets.py`) or procedural maps (`--set procedural`, or `--seeds 0 1 2`). Any configuration option can be overridden on the command line. Plots go to `plots/` and results to `results/`.

`scripts/experiments/` produces the comparisons for the paper:

* `compare_methods.py`: Reconstructs the same measurements with Gas Splatting and the grid-based baselines (SART, RBF-SART, LFD, LTD), and reports reconstruction error and time per case. The system matrix of the grid methods is timed apart from their solves, and runs are warmed up so that times are comparable.

* `num_measurements.py`: Reconstruction error of every method as measurements are added, pose by pose on GADEN scenarios (beam by beam on procedural maps). Meant for a simple environment, where the effect is easiest to see.

`scripts/figures/` produces the figures of the method:

* `injection_figure.py`: The Gaussian injection mechanism, step by step.

* `far_field_gradient.py`: Primitives far from unexplained gas receive no gradient, which motivates the injection.

* `densification_training.py`: Loss and error during training with the 3DGS densification and with the mass-conserving long-axis split, showing the loss spikes each one causes, and both reconstructions.

* `split_c_factor.py`: Error of the long-axis split as the children are placed farther from the parent center (the offset of half the major scale).

`scripts/tools/` holds tuning, checks and quick tests:

* `make_scenarios.py`: Extracts the GADEN scenarios of the tuning and held-out sets from the VGR house archives into `data/scenarios`.

* `benchmark.py`: Runs Gas Splatting on a set and reports, for every case, the normalized RMSE, recovered peak, mass error, gas inside obstacles, Gaussians and time, with a ground truth vs. reconstruction grid. Run it after any change to check that nothing got worse. Ablations are a single run each, e.g. `--original_dens` for the 3DGS densification.

* `sweep.py`: One-at-a-time hyperparameter sweep. Each parameter is varied on its own and gets a plot of every metric against its value. Progress is saved as runs finish, so an interrupted sweep resumes by running the same command again.

* `scale_invariance.py`: Checks that the reconstruction does not depend on the absolute concentration level, by repeating a run with the measurements multiplied by several factors.

* `noise_first.py`: Incremental run whose first batches only measure noise, checking that the concentration scale is raised when the gas is found.

* `test_obstacles.py`: Toy obstacle layouts on procedural maps, for quick checks of the obstacle penalty.

* `test_real_data.py`: Runs Gas Splatting on the real TDLAS sweeps in `real_data/`.

### GADEN scenarios and hyperparameter tuning

Hyperparameters are tuned on the `tuning` set and the final values are checked on the `held_out` set, made of other houses:

```bash
python scripts/tools/make_scenarios.py --vgr_dir ~/GADEN_files/scenarios   # once
python scripts/tools/benchmark.py --set tuning                              # current configuration

# 1. Learning rates (current value x 1/4 ... x8)
python scripts/tools/sweep.py --params concentration_lr pos_lr scale_lr
# 2. Densification and obstacle penalty
python scripts/tools/sweep.py --params gradient_threshold --values 0.01 0.02 0.04 0.08 0.16
python scripts/tools/sweep.py --params injection_scale --values 0.1 0.25 0.5 1 2
python scripts/tools/sweep.py --params obstacle_lambda --values 0 0.025 0.05 0.1 0.2 0.4
# 3. Stopping
python scripts/tools/sweep.py --params iterations early_stopping_patience early_stopping_min_delta

python scripts/tools/benchmark.py --set held_out                            # with the chosen values
```

After each stage, the chosen values are set in `config.py` (or passed on the command line) before sweeping the next one. Runs are parallel (`--workers`), so the times they report are only indicative; use `--workers 1` to measure them.

### Visualization

The `visualization/` directory contains components used to display training progress and standalone visualization tools:

* `visualization/live_visualizer.py`: Reusable live training visualizer instantiated by the trainer.

* `visualization/split_visualizer.py`: Standalone interactive tool for inspecting Gaussian splitting.

</details>


<details>
<summary><h2>⚙️ Configuration and CLI Options</h2></summary>

This project uses `simple_parsing` to dynamically handle configuration parameters from the command line. You can customize almost any aspect of the simulation, initialization, training, and densification processes.

To see a full list of available options for a given script, run:

```bash
python [script] --help
```

### Examples

Run a custom simulation with 50 beams and a maximum of 2000 training iterations:

```bash
python main.py --num_beams 50 --iterations 2000
```

Compare Gas Splatting with the baselines on three procedural maps, with 100 beams each:

```bash
python scripts/experiments/compare_methods.py --seeds 0 1 2 --num_beams 100
```

Train the model using custom learning rates and enable real-time visualization:

```bash
python main.py --pos_lr 0.005 --concentration_lr 0.002 --live_vis
```

> ⚠️ Important Note: Because configuration classes are shared across the entire project, some CLI options might not be applicable or will have no effect depending on the specific script you are executing. For example, `--seed` has no effect on scripts given `--seeds`, which set the seed of each procedural map.

</details>

## 📜 Acknowledgements and License

This project introduces custom implementations and novel features developed specifically for **Gas Splatting**. However, the core architecture and some of its utilities are deeply inspired by and built upon the open-source repository [r2_gaussian](https://github.com/ruyi-zha/r2_gaussian), which in turn is a derivative of the original 3D Gaussian Splatting implementation by Inria and MPII.

Because this repository contains derivative work, it is distributed under the identical **Gaussian-Splatting License**. 

* **The software is provided strictly for non-commercial, research, and evaluation purposes.**
* Any commercial use requires prior and explicit consent from the original licensors (Inria/MPII).

For more details, please refer to the `LICENSE` file included in this repository.