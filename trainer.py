import os
import math
import time
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import EnvironmentContext, MeasurementBatch
from utils.densification_utils import extract_candidate_positions, injection_scale
from utils.gaussian_utils import estimate_concentration_scale
from utils.obstacle_utils import DistanceFields, obstacle_penalty


@dataclass
class BatchResults:
    final_loss: float = 0.0
    batch_time: float = 0.0

@dataclass
class TrainingResults:
    loss_history: List[float] = field(default_factory=list)
    densify_history: Dict[int, dict] = field(default_factory=dict)
    rmse_history: Dict[int, float] = field(default_factory=dict)
    training_time: float = 0.0


def get_exp_lr_func(lr_init, lr_final, max_steps):
    def lr_func(step):
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0
        
        t = np.clip(step / max_steps, 0, 1)
        return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)

    return lr_func


class EarlyStopping:
    """Decides when to stop training a batch, from the losses of every iteration.

    Two rules, checked in this order:
    - Discrepancy principle (only with a known noise level, after densify_from): stop once
      the smoothed data loss is as large as the noise, since fitting further fits the noise.
    - Patience (after densify_until): stop once the smoothed total loss has not improved
      for early_stopping_patience iterations.
    """
    def __init__(self, cfg: Config, discrepancy: float | None = None):
        self.cfg = cfg
        self.discrepancy = discrepancy
        self.ema_loss = None
        self.best_ema_loss = float('inf')
        self.patience_counter = 0
        self.ema_data_loss = None

    def _ema(self, average: float | None, value: float) -> float:
        a = self.cfg.train.ema_alpha
        return value if average is None else (a * value) + ((1 - a) * average)

    def update(self, total_loss: float, data_loss: float, iteration: int) -> str | None:
        """Takes the losses of one iteration. Returns why training should stop, or None."""
        self.ema_loss = self._ema(self.ema_loss, total_loss)

        if self.discrepancy is not None:
            self.ema_data_loss = self._ema(self.ema_data_loss, data_loss)
            if iteration > self.cfg.densify.densify_from and self.ema_data_loss <= self.discrepancy:
                return f"Stopped at iteration {iteration}: the data residual reached the noise level"

        if iteration > self.cfg.densify.densify_until:
            # Relative improvement, so that the criterion does not depend on how small the loss is
            if self.ema_loss < self.best_ema_loss * (1 - self.cfg.train.early_stopping_min_delta):
                self.best_ema_loss = self.ema_loss
                self.patience_counter = 0  # Reset patience if we hit a new best
            else:
                self.patience_counter += 1 # Increment if it stalled or got worse

            if self.patience_counter >= self.cfg.train.early_stopping_patience:
                return f"Early stopping triggered at iteration {iteration} (EMA Loss stalled at {self.ema_loss:.5f})"

        return None


class Trainer:
    def __init__(self, model: GasSplattingModel, cfg: Config, environment: EnvironmentContext | None = None):
        self.model = model
        self.ground_truth = environment.ground_truth.gas_map if environment and environment.ground_truth else None
        # Evaluation only looks at free cells, where gas can actually be
        self.eval_mask = (~(environment.obstacles > 0.5)
                          if environment and environment.obstacles is not None and self.ground_truth is not None
                          else None)
        self.cfg = cfg
        if self.cfg.train.live_vis:
            from visualization.live_visualizer import LiveVisualizer
            self.visualizer = LiveVisualizer(cfg, environment)
        else:
            self.visualizer = None
        self.results = TrainingResults()
        self.global_iteration = 0

        # Prediction left by the last optimization step, reused by the injection
        # so it does not have to run a second forward pass over the whole buffer.
        self.last_prediction: torch.Tensor | None = None

        # Distance fields used by the obstacle penalty L_obs, None without an obstacle map
        self.distance_fields = DistanceFields.from_environment(environment, cfg)
        
        # Data Buffers
        dev = self.cfg.device
        self.buffer_beams = torch.empty((0, 2, 2), device=dev)
        self.buffer_measurements = torch.empty((0,), device=dev)
        self.buffer_weights = torch.empty((0,), device=dev)

        # Initial learning rate of every parameter group
        initial_lrs = {
            'pos': cfg.train.pos_lr,
            'scale': cfg.train.scale_lr,
            'rotation': cfg.train.rotation_lr,
            'concentration': cfg.train.concentration_lr,
        }
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam([
            {'params': [getattr(model, f'_{name}')], 'lr': lr, 'name': name} for name, lr in initial_lrs.items()
        ])

        # Every learning rate decays exponentially to a tenth of its initial value
        self.lr_funcs = {
            name: get_exp_lr_func(lr_init=lr, lr_final=0.1*lr, max_steps=cfg.train.iterations)
            for name, lr in initial_lrs.items()
        }

    def is_densify_it(self, iteration: int):
        return (iteration > self.cfg.densify.densify_from and 
                iteration < self.cfg.densify.densify_until and
                iteration % self.cfg.densify.densify_interval == 0)
    
    def update_learning_rates(self, iteration: int):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_funcs[param_group["name"]](iteration)

    def update_buffers(self, batch_data: MeasurementBatch):
        """Merges new incoming data into the training buffers"""
        # New measurements have maximum importance
        new_weights = torch.ones_like(batch_data.measurements)

        # Merge new data
        self.buffer_beams = torch.cat([self.buffer_beams, batch_data.beams], dim=0)
        self.buffer_measurements = torch.cat([self.buffer_measurements, batch_data.measurements], dim=0)
        self.buffer_weights = torch.cat([self.buffer_weights, new_weights], dim=0)

    def optimization_step(self, iteration: int) -> tuple[float, float]:
        """Executes a single optimization step (Forward + Backward + Step).
        Returns the total loss and the data loss alone."""
        self.optimizer.zero_grad()
        
        y_pred = self.model(self.buffer_beams)
        self.last_prediction = y_pred.detach()

        # Weighted measurements loss, relative to the concentration scale
        scale = self.model.scale
        data_loss = torch.mean(self.buffer_weights * torch.abs(y_pred - self.buffer_measurements)) / scale

        # Backward the data loss on its own so the densification gradient
        # accumulator only reflects gradients coming from the measurements,
        # not the obstacle penalization below.
        data_loss.backward()
        self.model.update_accum_gradient(self.buffer_beams)

        data_value = data_loss.item()
        total_loss = data_value

        # --- Obstacle penalty L_obs ---
        if self.distance_fields is not None and self.cfg.train.obstacle_lambda > 0.0:
            obstacle_loss = obstacle_penalty(self.model, self.distance_fields, self.cfg)

            # Backward separately so it never affects the densification accumulator
            obstacle_loss.backward()
            total_loss += obstacle_loss.item()
        # ----------------------

        self.update_learning_rates(iteration)
        self.optimizer.step()
        self.model.project_positions()

        return total_loss, data_value

    def inject_gaussians(self) -> int:
        """Injects new Gaussians into the regions the current model does not cover."""
        # Reuse the prediction of the last optimization step. It only lags behind by
        # one optimizer step and the densification that precedes this call, and both
        # the long-axis split and the pruning of negligible Gaussians leave the
        # predicted integrals essentially unchanged.
        if self.last_prediction is None or self.last_prediction.shape != self.buffer_measurements.shape:
            self.last_prediction = self.model(self.buffer_beams).detach()

        y_pred = self.last_prediction
        residuals = self.buffer_measurements - y_pred

        init_scale = injection_scale(self.cfg)

        final_pos = extract_candidate_positions(
            beams=self.buffer_beams,
            importance_scores=residuals,
            min_dist=init_scale,
            device=self.cfg.device,
            predictions=y_pred,
            existing_pos=self.model.get_pos().detach(),
        )

        num_injected = final_pos.shape[0]
        
        if num_injected > 0:
            self.model.inject_gaussians(
                optimizer=self.optimizer, 
                pos=final_pos, 
                scale=init_scale,
                concentration=self.cfg.densify.injection_concentration,
            )
            
        return num_injected

    def update_concentration_scale(self, batch_data: MeasurementBatch):
        """Raises the concentration scale when a batch exceeds it, so that it stays the largest
        path-averaged concentration measured so far. It never decreases, since earlier batches
        stay in the buffer."""
        batch_scale = estimate_concentration_scale(batch_data.beams, batch_data.measurements)
        current = self.model.concentration_scale
        if batch_scale <= 0 or (current is not None and batch_scale <= current):
            return

        self.model.set_concentration_scale(batch_scale)
        self.optimizer.state.clear()  # Adam moments refer to the old latents
        self.last_prediction = None
        if not self.cfg.quiet and current is not None:
            tqdm.write(f"Concentration scale raised from {current:.4g} to {batch_scale:.4g} ppm")

    def noise_discrepancy(self) -> float | None:
        """Data loss at which the residual is as large as the noise, None without a known noise level.

        Discrepancy principle: with a known noise level, stop once the data residual is as large as the
        noise, since fitting further fits the noise. A reading y with an error of std sigma*y has an
        expected absolute error of sqrt(2/pi)*sigma*y, so the data loss (mean absolute residual over
        the concentration scale) reaches the noise at this level.
        """
        if not self.cfg.train.noise_level:
            return None
        return (math.sqrt(2 / math.pi) * self.cfg.train.noise_level
                * float(self.buffer_measurements.abs().mean()) / float(self.model.scale))

    def _densify(self):
        """Splits and prunes the Gaussians, then injects new ones where the model falls short."""
        with torch.no_grad():
            stats = self.model.densify_and_prune(self.optimizer)
            stats['injections'] = self.inject_gaussians()

        self.results.densify_history[self.global_iteration] = stats

    def _evaluate(self) -> float:
        """RMSE of the current map against the ground truth, over the free cells."""
        current_map = self.model.render_map(cell_size=self.cfg.env.cell_size)
        error = current_map - self.ground_truth
        if self.eval_mask is not None:
            error = error[self.eval_mask]
        return float(np.sqrt(np.mean(error**2)))

    def train(self, batch_data: MeasurementBatch) -> BatchResults:
        self.update_concentration_scale(batch_data)
        self.update_buffers(batch_data)

        early_stopping = EarlyStopping(self.cfg, discrepancy=self.noise_discrepancy())

        # Training timer: optimization, early stopping and densification. Visualization
        # and evaluation are left out
        batch_time = 0.0

        current_loss = 0.0
        pbar = tqdm(range(self.cfg.train.iterations), desc="Training", dynamic_ncols=True, disable=self.cfg.quiet)        
        for iteration in pbar:
            t_start = time.time()
            
            current_loss, data_loss = self.optimization_step(iteration)
            self.results.loss_history.append(current_loss)

            stop_reason = early_stopping.update(current_loss, data_loss, iteration)

            # The iteration that stops training does not densify
            is_densifying = stop_reason is None and self.is_densify_it(iteration)
            if is_densifying:
                self._densify()

            batch_time += (time.time() - t_start)

            if stop_reason is not None:
                if not self.cfg.quiet:
                    tqdm.write(stop_reason)
                break

            if iteration % 100 == 0:
                pbar.set_postfix({'loss': f'{current_loss:.5f}', 'gaussians': self.model.num_gaussians})
            
            # --- Real time visualization ---
            if self.visualizer and (iteration % 20 == 0 or is_densifying):
                self.visualizer.update(
                    iteration=self.global_iteration, 
                    loss_history=self.results.loss_history, 
                    model=self.model
                )
            # ---------------------------------
            
            # --- Model evaluation (needs a ground truth) ---
            if self.cfg.train.do_eval and self.ground_truth is not None and iteration % self.cfg.train.eval_interval == 0:
                self.results.rmse_history[self.global_iteration] = self._evaluate()
            # ----------------------------

            self.global_iteration += 1

        pbar.close()

        self.results.training_time += batch_time

        return BatchResults(final_loss=current_loss, batch_time=batch_time)

    def finish(self, gif_path: str = "plots/training.gif") -> TrainingResults:
        """Handles post-training cleanup and returns the global training history."""
        if self.visualizer:
            if os.path.dirname(gif_path):
                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            self.visualizer.save_gif(gif_path)

        del self.buffer_beams
        del self.buffer_measurements
        del self.buffer_weights

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.results
