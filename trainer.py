import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import EnvironmentContext, MeasurementBatch
from utils.densification_utils import extract_candidate_positions, injection_scale
from utils.gaussian_utils import estimate_concentration_scale
from utils.obstacle_utils import compute_esdf, compute_esdf_gradient, compute_aasdf


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
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        
        t = np.clip(step / max_steps, 0, 1)
        return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)

    return lr_func

class Trainer:
    def __init__(self, model: GasSplattingModel, cfg: Config, environment: EnvironmentContext | None = None, max_buffer_size: int | None = None):
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

        # --- Distance fields used by the obstacle penalty L_obs ---
        # ESDF D, its gradient, and the axis-aligned SDF (D_x, D_y), all in meters
        self.esdf_tensor = None
        self.esdf_grad_tensor = None
        self.aasdf_tensor = None

        if environment and environment.obstacles is not None:
            esdf = compute_esdf(environment.obstacles, cell_size=cfg.env.cell_size)
            dev = self.cfg.device

            # (1, 1, H, W)
            self.esdf_tensor = torch.tensor(esdf, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)

            # (1, 2, H, W)
            self.esdf_grad_tensor = torch.tensor(
                compute_esdf_gradient(esdf), dtype=torch.float32, device=dev
            ).unsqueeze(0)

            # (1, 2, H, W): D_x and D_y
            self.aasdf_tensor = torch.tensor(
                compute_aasdf(
                    environment.obstacles,
                    cell_size=cfg.env.cell_size,
                    max_dist=float(max(cfg.env.map_size)),
                ),
                dtype=torch.float32, device=dev
            ).unsqueeze(0)
        # -------------------------------------------------------------------
        
        # Data Buffers
        self.max_buffer_size = max_buffer_size or float('inf')
        dev = self.cfg.device
        self.buffer_beams = torch.empty((0, 2, 2), device=dev)
        self.buffer_measurements = torch.empty((0,), device=dev)
        self.buffer_weights = torch.empty((0,), device=dev)

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam([
            {'params': [model._pos], 'lr': self.cfg.train.pos_lr, 'name': 'pos'},
            {'params': [model._scale], 'lr': self.cfg.train.scale_lr, 'name': 'scale'},
            {'params': [model._rotation], 'lr': self.cfg.train.rotation_lr, 'name': 'rotation'},
            {'params': [model._concentration], 'lr': self.cfg.train.concentration_lr, 'name': 'concentration'},
        ])

        # Setup Learning Rates Schedulers
        self.pos_lr_func = get_exp_lr_func(
            lr_init=cfg.train.pos_lr,
            lr_final=0.1*cfg.train.pos_lr,
            max_steps=cfg.train.iterations
        )
        self.scale_lr_func = get_exp_lr_func(
            lr_init=cfg.train.scale_lr,
            lr_final=0.1*cfg.train.scale_lr,
            max_steps=cfg.train.iterations
        )
        self.rotation_lr_func = get_exp_lr_func(
            lr_init=cfg.train.rotation_lr,
            lr_final=0.1*cfg.train.rotation_lr,
            max_steps=cfg.train.iterations
        )
        self.concentration_lr_func = get_exp_lr_func(
            lr_init=cfg.train.concentration_lr,
            lr_final=0.1*cfg.train.concentration_lr,
            max_steps=cfg.train.iterations
        )

    def is_densify_it(self, iteration: int):
        return (iteration > self.cfg.densify.densify_from and 
                iteration < self.cfg.densify.densify_until and
                iteration % self.cfg.densify.densify_interval == 0)
    
    def update_learning_rates(self, iteration: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pos":
                param_group["lr"] = self.pos_lr_func(iteration)
            elif param_group["name"] == "scale":
                param_group["lr"] = self.scale_lr_func(iteration)
            elif param_group["name"] == "rotation":
                param_group["lr"] = self.rotation_lr_func(iteration)
            elif param_group["name"] == "concentration":
                param_group["lr"] = self.concentration_lr_func(iteration)

    def update_buffers(self, batch_data: MeasurementBatch):
        """Merges new incoming data into the training buffers"""
        # New measurements have maximum importance
        new_weights = torch.ones_like(batch_data.measurements)

        # Merge new data
        self.buffer_beams = torch.cat([self.buffer_beams, batch_data.beams], dim=0)
        self.buffer_measurements = torch.cat([self.buffer_measurements, batch_data.measurements], dim=0)
        self.buffer_weights = torch.cat([self.buffer_weights, new_weights], dim=0)

        # Truncate buffers for memory safety
        if self.buffer_measurements.shape[0] > self.max_buffer_size:
            self.buffer_beams = self.buffer_beams[-self.max_buffer_size:]
            self.buffer_measurements = self.buffer_measurements[-self.max_buffer_size:]
            self.buffer_weights = self.buffer_weights[-self.max_buffer_size:]

    def optimization_step(self, iteration: int):
        """Executes a single optimization step (Forward + Backward + Step)."""
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

        total_loss = data_loss.item()

        # --- Obstacle penalty L_obs (ESDF and AA-SDF terms, complementary) ---
        use_obstacles = (self.esdf_tensor is not None
                         and self.esdf_grad_tensor is not None
                         and self.aasdf_tensor is not None
                         and self.cfg.train.obstacle_lambda > 0.0)

        if use_obstacles:
            pos = self.model.get_pos()                      # (K, 2)
            Sigma = self.model.get_covariance()             # (K, 2, 2)
            concentrations = self.model.get_concentration() # (K,)

            # Normalize positions to [-1, 1] for grid sampling
            map_w, map_h = self.model.map_size[0], self.model.map_size[1]
            norm_x = (pos[:, 0] / map_w) * 2.0 - 1.0
            norm_y = (pos[:, 1] / map_h) * 2.0 - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2) # (1, 1, K, 2)

            # --- ESDF term ---
            # D(mu_k): (1, 1, 1, K) -> (K,)
            esdf_at_mu = F.grid_sample(
                self.esdf_tensor, grid,
                mode='bilinear', padding_mode='border', align_corners=False
            ).view(-1)

            # grad D(mu_k): (1, 2, 1, K) -> (K, 2)
            esdf_grad_at_mu = F.grid_sample(
                self.esdf_grad_tensor, grid,
                mode='bilinear', padding_mode='border', align_corners=False
            ).squeeze(0).squeeze(1).transpose(0, 1)

            # Unit normal n_k pointing away from the nearest surface
            n = F.normalize(esdf_grad_at_mu, p=2, dim=-1, eps=1e-6) # (K, 2)

            # Variance along the normal, n^T Sigma n: (K, 1, 2) @ (K, 2, 2) @ (K, 2, 1) -> (K,)
            normal_variance = torch.bmm(
                n.unsqueeze(1), torch.bmm(Sigma, n.unsqueeze(-1))
            ).view(-1)
            sigma_n = torch.sqrt(torch.clamp(normal_variance, min=1e-7))

            esdf_violation = F.relu(self.cfg.train.obstacle_sigma_k * sigma_n - esdf_at_mu)

            # --- AA-SDF term ---
            # (D_x(mu_k), D_y(mu_k)): (1, 2, 1, K) -> (K, 2)
            aasdf_at_mu = F.grid_sample(
                self.aasdf_tensor, grid,
                mode='bilinear', padding_mode='border', align_corners=False
            ).squeeze(0).squeeze(1).transpose(0, 1)

            # Along the coordinate axes the variance reduces to the diagonal of Sigma: (K, 2)
            sigma_axis = torch.sqrt(torch.clamp(torch.diagonal(Sigma, dim1=1, dim2=2), min=1e-7))

            aasdf_violation = F.relu(self.cfg.train.obstacle_sigma_k * sigma_axis - aasdf_at_mu)

            # Weight by sg(rho_k): keeps the penalty commensurate with the data
            # term, and stops fading a Gaussian out from dodging it.
            obstacle_loss = self.cfg.train.obstacle_lambda * torch.sum(
                (esdf_violation + aasdf_violation.sum(dim=-1)) * concentrations.detach()
            ) / scale

            # Backward separately so it never affects the densification accumulator
            obstacle_loss.backward()
            total_loss += obstacle_loss.item()
        # ----------------------

        self.update_learning_rates(iteration)
        self.optimizer.step()
        self.model.project_positions()

        return total_loss

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

    def train(self, batch_data: MeasurementBatch) -> BatchResults:
        self.update_concentration_scale(batch_data)
        self.update_buffers(batch_data)

        # Early stopping init
        ema_loss = None
        best_ema_loss = float('inf')
        patience_counter = 0

        # Training timer
        batch_time = 0.0

        current_loss = 0.0
        pbar = tqdm(range(self.cfg.train.iterations), desc="Training", dynamic_ncols=True, disable=self.cfg.quiet)        
        for iteration in pbar:
            t_start = time.time()
            
            current_loss = self.optimization_step(iteration)
            self.results.loss_history.append(current_loss)

            # --- Early stopping ---
            # Update the smoothed EMA loss
            if ema_loss is None:
                ema_loss = current_loss
            else:
                ema_loss = (self.cfg.train.ema_alpha * current_loss) + ((1 - self.cfg.train.ema_alpha) * ema_loss)

            if iteration > self.cfg.densify.densify_until:
                # Check for significant improvement
                # Relative improvement, so that the criterion does not depend on how small the loss is
                if ema_loss < best_ema_loss * (1 - self.cfg.train.early_stopping_min_delta):
                    best_ema_loss = ema_loss
                    patience_counter = 0  # Reset patience if we hit a new best
                else:
                    patience_counter += 1 # Increment if it stalled or got worse
                    
                # Halt if patience runs out
                if patience_counter >= self.cfg.train.early_stopping_patience:
                    batch_time += (time.time() - t_start) # Add iteration time before breaking
                    if not self.cfg.quiet:
                        tqdm.write(f"Early stopping triggered at iteration {iteration} (EMA Loss stalled at {ema_loss:.5f})")
                    break 
            # ----------------------------

            # --- Densification ---
            is_densifying = self.is_densify_it(iteration)
            if is_densifying:
                with torch.no_grad():
                    stats = self.model.densify_and_prune(self.optimizer)
                    stats['injections'] = self.inject_gaussians()

                self.results.densify_history[self.global_iteration] = stats
            # -------------------------

            batch_time += (time.time() - t_start)

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
                current_map = self.model.render_map(cell_size=self.cfg.env.cell_size)
                error = current_map - self.ground_truth
                if self.eval_mask is not None:
                    error = error[self.eval_mask]
                rmse = float(np.sqrt(np.mean(error**2)))
                self.results.rmse_history[self.global_iteration] = rmse
            # ----------------------------

            self.global_iteration += 1

        pbar.close()

        self.results.training_time += batch_time

        return BatchResults(final_loss=current_loss, batch_time=batch_time)

    def finish(self, gif_path: str = "plots/training.gif") -> TrainingResults:
        """Handles post-training cleanup and returns the global training history."""
        if self.visualizer:
            os.makedirs("plots", exist_ok=True)
            self.visualizer.save_gif(gif_path)

        del self.buffer_beams
        del self.buffer_measurements
        del self.buffer_weights

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.results