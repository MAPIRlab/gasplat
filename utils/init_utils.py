import time
import torch
from dataclasses import dataclass, field
from typing import Any, Dict

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import MeasurementBatch
from utils.densification_utils import extract_candidate_positions, injection_scale
from utils.gaussian_utils import estimate_concentration_scale

@dataclass
class InitializationData:
    """Data class to hold the initialization results"""
    pos: torch.Tensor
    concentration: torch.Tensor
    std: torch.Tensor
    # Optional dictionary to store additional information about the initialization process
    info: Dict[str, Any] = field(default_factory=dict) 

def injection_initialization(batch: MeasurementBatch, cfg: Config):
    """
    Initializes Gaussians based on high-concentration beams using Continuous Spatial NMS.
    """
    init_std_val = injection_scale(cfg)
    
    final_pos = extract_candidate_positions(
        beams=batch.beams,
        importance_scores=batch.measurements,
        min_dist=init_std_val,
        device=cfg.device
    )
    
    num_final = final_pos.shape[0]
    
    if num_final == 0:
        return InitializationData(
            pos=torch.empty((0, 2), device=cfg.device),
            concentration=torch.empty((0,), device=cfg.device),
            std=torch.empty((0,), device=cfg.device)
        )
        
    std = torch.full((num_final,), init_std_val, dtype=torch.float32, device=cfg.device)
    # Same value the injection uses, so initialization is the injection applied
    # to an empty model rather than a separate procedure
    concentration = torch.full((num_final,), cfg.densify.injection_concentration, dtype=torch.float32, device=cfg.device)
    
    return InitializationData(pos=final_pos, concentration=concentration, std=std)

def setup_gs_model(batch: MeasurementBatch, cfg: Config) -> tuple[GasSplattingModel, InitializationData]:
    """
    Initializes Gas Splatting model.

    Args:
        batch (MeasurementBatch): Batch containing beams geometry and their measurements.
        cfg (Config): Configuration object with model and other parameters.

    Returns:
        tuple[GasSplattingModel, InitializationData]: Initialized Gas Splatting model and the initialization data,
        whose info["setup_time"] is the time this initialization took (s).
    """
    t_start = time.time()
    init_data = injection_initialization(batch, cfg)
        
    initial_gaussians = init_data.pos.shape[0]

    # Set the scale first: initial concentrations are relative to it
    batch_scale = estimate_concentration_scale(batch.beams, batch.measurements)

    if initial_gaussians > 0:
        model = GasSplattingModel(initial_gaussians, cfg).to(cfg.device)
        if batch_scale > 0:
            model.set_concentration_scale(batch_scale)
        model.initialize_gaussians(
            init_data.pos.to(cfg.device), 
            init_data.concentration.to(cfg.device), 
            init_data.std.to(cfg.device)
        )
    else:
        model = GasSplattingModel(1, cfg).to(cfg.device)
        if batch_scale > 0:
            model.set_concentration_scale(batch_scale)
        init_data.pos = model.get_pos().detach().cpu()

    if model._pos.is_cuda:
        torch.cuda.synchronize(model._pos.device)  # GPU work is asynchronous; include it here
    init_data.info["setup_time"] = time.time() - t_start
    return model, init_data