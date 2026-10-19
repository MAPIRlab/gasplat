"""
Runs a GADEN simulation with gaden_core and writes a GASPLAT scenario package,
following the same principles as the VGR scenarios (utils/gaden_utils.py): one
snapshot once the plume is steady, the occupancy layer at a height above the
floor, cropped to the free space.

Besides the ground truth at the cell centers (gas_map, used for evaluation),
the package holds the same slice sampled on a finer grid (gas_map_fine, fine_factor
samples per cell side). The measurements are simulated on the fine map, so that
the grid-based baselines do not reconstruct on the very grid the data was
generated on (the "inverse crime" of tomography).

    python scripts/tools/simulate_gaden.py ~/GADEN_files/test_env/scenarios/10x6_empty_room
        [--config config1] [--simulation sim1 | --sim_dir gaden_sims/10x6_empty_room/center_plume]
        [--time 1000 [300 ...]] [--height 0.85 | source] [--fine_factor 4]

--sim_dir runs a simulation kept in this repo (gaden_sims/) instead of one of the
scenario: a folder with its sim.yaml and, if it changes the environment
configuration (e.g. still air), a config.yaml with the files it refers to. The
folder name names the simulation. --time takes several snapshot times, written
as separate packages from a single run.

--height is the beam plane above the floor (m), or "source" for the layer that
contains the gas source (the upper one if the source lies on a layer boundary).

Needs the gaden_core Python bindings (from gaden_py import gaden), see the README
of gaden_core. The scenario folder is copied to a temporary directory, since
preprocessing writes into it.
"""
import os
import re
import sys
import json
import shutil
import argparse
import tempfile
import subprocess
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.gaden_utils import DEFAULT_ROOT, FREE, OBSTACLE, OUTLET, save_preview

GADEN_CORE = os.path.expanduser("~/gaden_core")


def gaden_core_version() -> str:
    try:
        commit = subprocess.run(["git", "-C", GADEN_CORE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", GADEN_CORE, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + (" (local changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def install_sim_dir(work: str, config: str, sim_dir: str) -> tuple[str, str]:
    """Copies a repo simulation into the working copy; returns its configuration and simulation names."""
    simulation = os.path.basename(os.path.normpath(sim_dir))
    if os.path.exists(os.path.join(sim_dir, "config.yaml")):
        config = simulation
        shutil.copytree(sim_dir, os.path.join(work, "environment_configurations", config),
                        ignore=shutil.ignore_patterns("sim.yaml"))
    target = os.path.join(work, "environment_configurations", config, "simulations", simulation)
    os.makedirs(target, exist_ok=True)
    shutil.copy(os.path.join(sim_dir, "sim.yaml"), target)
    return config, simulation


def simulate(scenario_dir: str, config: str, simulation: str, times_s: list[float], height: float | str,
             fine_factor: int, root: str, sim_dir: str | None = None, crop_margin: int = 1) -> list[tuple[str, dict]]:
    from gaden_py import gaden

    scenario = os.path.basename(os.path.normpath(scenario_dir))
    packages = []
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, scenario)
        shutil.copytree(scenario_dir, work)
        if sim_dir:
            config, simulation = install_sim_dir(work, config, sim_dir)
        config_dir = os.path.join(work, "environment_configurations", config)
        with open(os.path.join(config_dir, "config.yaml")) as f:
            config_yaml = f.read()
        with open(os.path.join(config_dir, "simulations", simulation, "sim.yaml")) as f:
            sim_yaml = f.read()

        metadata = gaden.EnvironmentConfigMetadata(config_dir)
        metadata.ReadDirectory()
        env_config = gaden.Preprocessing.Preprocess(metadata)
        params = metadata.GetSimulationParams(simulation)
        params.saveResults = False
        params.expectedNumIterations = int(max(times_s) / params.deltaTime) + 1
        sim = gaden.RunningSimulation(params, env_config)

        # --- occupancy: same codes as VGR (0 free, 1 obstacle, 2 outlet) ---
        env = env_config.environment
        d = env.description
        nx, ny, nz = d.dimensions.x, d.dimensions.y, d.dimensions.z
        cs = round(float(d.cellSize), 6)  # stored as float32
        states = {gaden.Environment.CellState.Free: FREE, gaden.Environment.CellState.Obstacle: OBSTACLE,
                  gaden.Environment.CellState.Outlet: OUTLET}
        cells = np.empty((nx, ny, nz), dtype=np.uint8)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    cells[i, j, k] = states.get(env.at(gaden.Vector3i(i, j, k)), OBSTACLE)
        env_min = np.array([d.minCoord.x, d.minCoord.y, d.minCoord.z], dtype=float)
        floor = env_min[2] + int(np.argmax((cells == FREE).reshape(-1, nz).any(axis=0))) * cs
        position = re.search(r"position:\s*\[([^\]]+)\]", sim_yaml)
        source = [float(v) for v in position.group(1).split(",")] if position else [float("nan")] * 3
        target = source[2] if height == "source" else floor + float(height)
        k = int(np.floor((target - env_min[2]) / cs + 1e-6))
        z = env_min[2] + (k + 0.5) * cs

        layer = cells[:, :, k]
        free_xy = np.argwhere(layer == FREE)
        lo = np.maximum(free_xy.min(axis=0) - crop_margin, 0)
        hi = np.minimum(free_xy.max(axis=0) + crop_margin + 1, layer.shape)
        layer = layer[lo[0]:hi[0], lo[1]:hi[1]]
        W, H = layer.shape

        occupancy = np.ascontiguousarray(layer.T)
        origin_world = env_min[:2] + lo * cs

        # --- concentration at the cell centers, and f x f samples per cell ---
        def sample(fine: int) -> np.ndarray:
            offsets = (np.arange(fine) + 0.5) / fine
            out = np.zeros((H * fine, W * fine), dtype=np.float32)
            for c in range(W):
                for r in range(H):
                    if layer[c, r] != FREE:
                        continue
                    x0 = env_min[0] + (lo[0] + c) * cs
                    y0 = env_min[1] + (lo[1] + r) * cs
                    for a, oy in enumerate(offsets):
                        for b, ox in enumerate(offsets):
                            out[r * fine + a, c * fine + b] = sim.SampleConcentration(
                                gaden.Vector3(x0 + ox * cs, y0 + oy * cs, z))
            return out

        for time_s in sorted(times_s):
            while sim.GetCurrentTime() < time_s:
                sim.AdvanceTimestep()
            gas_map, gas_map_fine = sample(1), sample(fine_factor)
            name = f"{scenario.lower()}_{simulation}_h{z - floor:.2f}_t{time_s:.0f}"
            meta = {
                "name": name,
                "generator": "gaden_core",
                "gaden_core_commit": gaden_core_version(),
                "scenario_dir": os.path.abspath(scenario_dir),
                "scenario": scenario,
                "configuration": config,
                "simulation": simulation,
                "sim_dir": os.path.relpath(sim_dir) if sim_dir else None,
                "time_s": time_s,
                "num_filaments": int(sim.GetFilaments().size()),
                "cell_size": cs,
                "fine_factor": fine_factor,
                "map_size_m": [W * cs, H * cs],
                "grid_shape_hw": [H, W],
                "origin_world": origin_world.tolist(),
                "layer_index": k,
                "layer_center_z": z,
                "height_above_floor": z - floor,
                "source_world": source,
                "source_local": [source[0] - origin_world[0], source[1] - origin_world[1]],
                "sim_yaml": sim_yaml,
                "config_yaml": config_yaml,
                "units": "ppm (maps), ppm*m (beam integrals)",
                "occupancy_codes": {"free": FREE, "obstacle": OBSTACLE, "outlet": OUTLET},
            }

            out_dir = os.path.join(root, name)
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(os.path.join(out_dir, "scenario.npz"), gas_map=gas_map, gas_map_fine=gas_map_fine,
                                occupancy=occupancy)
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            save_preview(os.path.join(out_dir, "preview.png"), gas_map, occupancy, meta)
            packages.append((out_dir, meta))
    return packages


def main():
    parser = argparse.ArgumentParser(description="Simulate a GADEN scenario and extract a GASPLAT scenario package")
    parser.add_argument("scenario_dir", help="GADEN scenario folder (cad_models, wind_simulations, environment_configurations)")
    parser.add_argument("--config", default="config1", help="Environment configuration")
    parser.add_argument("--simulation", default="sim1", help="Simulation within the configuration")
    parser.add_argument("--sim_dir", default=None, help="Simulation kept in this repo (gaden_sims/...), instead of --simulation")
    parser.add_argument("--time", type=float, nargs="+", default=[1000.0], help="Simulated times of the snapshots (s)")
    parser.add_argument("--height", default="0.85", help='Beam plane above the floor (m), or "source"')
    parser.add_argument("--fine_factor", type=int, default=4, help="Samples per cell side of the fine map")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Directory for the scenario package")
    args = parser.parse_args()

    height = args.height if args.height == "source" else float(args.height)
    for out_dir, meta in simulate(args.scenario_dir, args.config, args.simulation, args.time, height,
                                  args.fine_factor, args.root, args.sim_dir):
        print(f"Scenario written to {out_dir}")
        print(json.dumps({k: meta[k] for k in ("grid_shape_hw", "map_size_m", "height_above_floor", "num_filaments",
                                               "fine_factor", "gaden_core_commit")}, indent=2))


if __name__ == "__main__":
    main()
