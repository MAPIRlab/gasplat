"""
Checks that GASPLAT does not depend on the absolute concentration level: the
same measurements, multiplied by different factors, must give the same
reconstruction up to that factor (same Gaussians, same normalized error).
Concentrations are learned relative to a scale taken from the measurements,
so any difference beyond float rounding means a hyperparameter or threshold
has become absolute again.

    python scripts/scale_invariance.py [--scenario DIR] [--factors 0.01 1 100]

Without --scenario, the first scenario of the tuning set and a procedural map
are tested.
"""
import os
import sys
import copy
import numpy as np
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.scenario_sets import scenario_dirs
from utils.experiment_utils import gas_map_metrics
from utils.sim_utils import EnvironmentContext, GroundTruth, MeasurementBatch, generate_simulation_data

MAP_TOLERANCE = 0.02     # max |map / factor - reference map|, relative to the true peak
NRMSE_TOLERANCE = 0.005  # max NRMSE difference (absolute, 0.5 percentage points)


def check(cfg: Config, factors: list[float], label: str) -> bool:
    cfg = copy.deepcopy(cfg)
    batch, env = generate_simulation_data(cfg)  # also sets the map and cell size of the scenario
    gt = env.ground_truth.gas_map
    print(f"\n== {label}")
    reference, ok = None, True
    for f in factors:
        run_cfg = copy.deepcopy(cfg)
        scaled_batch = MeasurementBatch(batch.beams.clone(), batch.measurements * f, batch.pose_ids)
        scaled_env = EnvironmentContext(env.obstacles, GroundTruth(gt * f, env.ground_truth.y_true * f))
        model, _ = setup_gs_model(scaled_batch, run_cfg)
        trainer = Trainer(model, run_cfg, environment=scaled_env)
        trainer.train(scaled_batch)
        iterations = len(trainer.finish().loss_history)
        pred = model.render_map(run_cfg.env.cell_size) / f
        m = gas_map_metrics(gt, pred, env.obstacles)
        if reference is None:
            reference = (pred, m, model.num_gaussians)
        diff = np.abs(pred - reference[0]).max() / gt.max()
        same = (model.num_gaussians == reference[2] and diff <= MAP_TOLERANCE
                and abs(m["nrmse"] - reference[1]["nrmse"]) <= NRMSE_TOLERANCE)
        ok &= same
        print(f"x{f:<8g} scale {model.concentration_scale:10.4g}  gaussians {model.num_gaussians:4d}  "
              f"iterations {iterations:5d}  NRMSE {100 * m['nrmse']:6.2f}%  "
              f"max map difference {100 * diff:5.2f}% of peak  {'ok' if same else 'MISMATCH'}")
    return ok


def main():
    parser = ArgumentParser(description="Check that results do not depend on the concentration level")
    parser.add_arguments(Config, dest="cfg")
    parser.add_argument("--factors", nargs="+", type=float, default=[0.01, 1, 100])
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    # Float32 rounding differs between factors and shifts the iteration where early
    # stopping triggers; a fixed number of iterations isolates the scale dependence
    cfg.train.early_stopping_patience = cfg.train.iterations

    cases = []
    if cfg.sim.scenario:
        cases.append((cfg, os.path.basename(os.path.normpath(cfg.sim.scenario))))
    else:
        scenario_cfg = copy.deepcopy(cfg)
        scenario_cfg.sim.scenario = scenario_dirs("tuning")[0]
        cases += [(scenario_cfg, os.path.basename(scenario_cfg.sim.scenario)), (cfg, "procedural map")]

    results = [check(c, args.factors, label) for c, label in cases]
    print("\nPASS" if all(results) else "\nFAIL: results depend on the concentration level")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
