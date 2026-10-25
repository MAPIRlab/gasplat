"""
Time spent in each part of GASPLAT's training on one case, per iteration: the
forward pass (pair selection, integrals), both backward passes (data loss and
obstacle penalty), the optimizer, the densification and so on.

It wraps the functions a training step calls (it does not copy the trainer's
code, so a renamed function makes it fail loudly) and times each one between
device syncs, so the time of a part includes the GPU finishing its work. Nested
parts are reported without their children. The syncs make the total somewhat
longer than an uninstrumented run on the GPU; on the CPU they cost nothing.

    python scripts/tools/profile_step.py --scenario data/scenarios/house14_1,2-4,5_fast_h0.85_it1999
    python scripts/tools/profile_step.py --scenario ... --device_type cuda
    python scripts/tools/profile_step.py --seed 0 --threads 1      # procedural map
"""
import os
import sys
import time
import functools
import collections
import torch
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import gs_model
import trainer as trainer_module
from config import Config
from utils.scenario_sets import case_name
from utils.sim_utils import generate_simulation_data
from utils.init_utils import setup_gs_model

totals, calls, stack = collections.defaultdict(float), collections.Counter(), []


def instrument(sync):
    """Wraps the parts of a training step so that each call adds its time to `totals`."""
    def timed(label, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            sync()
            t0 = time.perf_counter()
            stack.append(0.0)
            out = f(*args, **kwargs)
            sync()
            elapsed = time.perf_counter() - t0
            children = stack.pop()
            totals[label] += elapsed - children
            calls[label] += 1
            if stack:
                stack[-1] += elapsed
            return out
        return wrapper

    def patch(owner, name, label):
        setattr(owner, name, timed(label, getattr(owner, name)))

    M, T = gs_model.GasSplattingModel, trainer_module.Trainer
    # A step calls backward on the data loss, then on the obstacle penalty if the map has obstacles
    backward, order = torch.Tensor.backward, collections.Counter()

    def labeled_backward(self, *args, **kwargs):
        order["calls"] += 1
        label = "backward of the data loss" if order["calls"] == 1 else "backward of the obstacle penalty"
        return timed(label, backward)(self, *args, **kwargs)
    torch.Tensor.backward = labeled_backward

    step = T.optimization_step

    def counted_step(self, *args, **kwargs):
        order["calls"] = 0
        return step(self, *args, **kwargs)
    T.optimization_step = counted_step
    patch(T, "optimization_step", "step: rest (loss, obstacle term arithmetic)")
    patch(M, "forward", "forward: rest (nonzero, gathers)")
    patch(gs_model, "segment_mahalanobis_sq", "forward: pair selection (N x K distances)")
    patch(gs_model, "compute_sparse_integral", "forward: closed-form integrals of the pairs")
    patch(M, "get_covariance_inverse", "covariance inverse (R S^-2 R^T)")
    patch(M, "get_covariance", "covariance (R S^2 R^T, obstacle term)")
    patch(trainer_module.F, "grid_sample", "obstacle term: distance field lookups")
    patch(M, "update_accum_gradient", "densification statistic accumulation")
    patch(torch.optim.Adam, "step", "Adam step")
    patch(M, "project_positions", "projection onto the map")
    patch(T, "update_learning_rates", "learning rate schedule")
    patch(M, "densify_and_prune", "densification: split and prune")
    patch(T, "inject_gaussians", "densification: injection")


def main():
    parser = ArgumentParser(description="Time spent in each part of a GASPLAT training step")
    parser.add_arguments(Config, dest="cfg")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads of PyTorch")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.quiet = True
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    torch.set_num_threads(args.threads)

    batch, environment = generate_simulation_data(cfg)

    def train():
        model, _ = setup_gs_model(batch, cfg)
        trainer = trainer_module.Trainer(model, cfg, environment=environment)
        trainer.train(batch)
        return trainer.results, model

    train()   # warm-up, not timed
    instrument(torch.cuda.synchronize if cfg.device.type == "cuda" else (lambda: None))
    results, model = train()

    iterations = len(results.loss_history)
    total = sum(totals.values())
    print(f"{case_name(cfg)} on {cfg.device.type} ({torch.get_num_threads()} threads): {iterations} iterations, "
          f"{model.num_gaussians} Gaussians at the end, {len(batch.measurements)} beams")
    print(f"Training: {results.training_time:.2f} s, {1000 * results.training_time / iterations:.2f} ms per iteration")
    for label, t in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"  {label:46s} {1000 * t / iterations:6.3f} ms/it {100 * t / total:5.1f}%  ({calls[label]} calls)")


if __name__ == "__main__":
    main()
