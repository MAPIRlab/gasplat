"""
Checks that GASPLAT does not depend on the absolute concentration level: the
same measurements, multiplied by different factors, must give the same
reconstruction up to that factor (same Gaussians, same normalized error).
Concentrations are learned relative to a scale taken from the measurements,
so any difference beyond float rounding means a hyperparameter or threshold
has become absolute again.

Training is chaotic over long runs: the L1 loss flips the sign of a beam's
push when its residual crosses zero, and densification makes discrete split
decisions, so float rounding alone ends up in a different set of Gaussians
after many iterations, at any scale. The check therefore runs on the CPU,
whose rounding is smallest, and only for a short run that includes the first
densification step (--iterations, 200 by default); a scale dependence shows up
there as a large difference, while rounding stays far below the tolerance.

    python scripts/tools/scale_invariance.py [--scenario DIR] [--factors 0.01 1 100]

Without --scenario, the first scenario of the tuning set and a procedural map
are tested.
"""
import os
import sys
import copy
import numpy as np
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.scenario_sets import scenario_dirs
from utils.experiment_utils import gas_map_metrics
from utils.sim_utils import EnvironmentContext, GroundTruth, MeasurementBatch, generate_simulation_data

MAP_TOLERANCE = 1e-3     # max |map / factor - reference map|, relative to the true peak
NRMSE_TOLERANCE = 1e-4   # max NRMSE difference (absolute)


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
    cfg.device_type = "cpu"
    if "--iterations" not in sys.argv:
        cfg.train.iterations = 200
    # No early stopping: a fixed number of iterations for every factor
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
