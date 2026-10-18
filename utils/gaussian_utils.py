import math
import torch

def inverse_softplus(x):
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))

def compute_definite_integral(pos, cov_inv, concentration, beams):
    """
    Computes predicted projections using the definite integral over line segments,
    given gaussians' parameters and rays defined by a single 'beams' tensor.
    
    Args:
        pos (Tensor (K,2)): Gaussians' positions (mu)
        cov_inv (Tensor (K,2,2)): Gaussians' covariance inverse matrix (Sigma^-1)
        concentration (Tensor (K,)): Gaussians' concentrations (rho)
        beams (Tensor (N, 2, 2)): Rays defined by pairs of [start_point, end_point]

    Returns:
        y_pred (Tensor (N,)): Predicted projections
    """

    # Extract starting (S) and ending (E) points 
    start_rays = beams[:, 0, :] # (N, 2)
    end_rays = beams[:, 1, :]   # (N, 2)

    # Ray vector v = E - S
    v_rays = end_rays - start_rays
    ray_lengths = torch.norm(v_rays, dim=1) # (N,)

    # d = S - mu
    # start_rays (N, 1, 2) - pos (1, K, 2) = d (N, K, 2)
    diff = start_rays.unsqueeze(1) - pos.unsqueeze(0)
    
    # A = 0.5 * v^T * Sigma^-1 * v
    A = 0.5 * torch.einsum('ni,kij,nj->nk', v_rays, cov_inv, v_rays)
    A = torch.clamp(A, min=1e-8)

    # B = v^T * Sigma^-1 * d
    B = torch.einsum('ni,kij,nkj->nk', v_rays, cov_inv, diff)

    # C = 0.5 * d^T * Sigma^-1 * d
    C = 0.5 * torch.einsum('nki,kij,nkj->nk', diff, cov_inv, diff)

    # exp(B^2 / 4A - C)
    term_exp = torch.exp((B**2) / (4 * A) - C)
    
    # Integration limits after variable change u = sqrt(A) * (t + B/(2*A))
    sqrt_A = torch.sqrt(A)
    u0 = B / (2 * sqrt_A)
    u1 = sqrt_A + u0
    
    # erf(u1) - erf(u0)
    erf_term = torch.erf(u1) - torch.erf(u0)
    
    # Integral from t=0 to t=1
    # sqrt(pi / 4A) = sqrt(pi) / (2 * sqrt_A)
    integral_t = (math.sqrt(math.pi) / (2 * sqrt_A)) * term_exp * erf_term # (N, K)

    # Multiply by beam length
    integral_matrix = integral_t * ray_lengths.unsqueeze(1) # (N, K) * (N, 1) -> (N, K)

    # Multiply by Gaussian concentration (rho)
    y_pred = torch.matmul(integral_matrix, concentration) # (N,)

    return y_pred

def segment_mahalanobis_sq(pos, cov_inv, beams):
    """
    Squared Mahalanobis distance between each beam segment and each Gaussian center,
    at the closest point of the segment, (N, K). Along a beam the squared distance is
    a*t^2 + b*t + c for t in [0, 1], minimized in closed form.
    """
    origin, direction = beams[:, 0, :], beams[:, 1, :] - beams[:, 0, :]   # (N, 2)
    offset = origin[:, None, :] - pos[None, :, :]                           # (N, K, 2)
    cov_inv_dir = torch.einsum('kij,nj->nki', cov_inv, direction)          # (N, K, 2)
    a = (direction[:, None, :] * cov_inv_dir).sum(-1)
    b = 2 * (offset * cov_inv_dir).sum(-1)
    c = torch.einsum('nki,kij,nkj->nk', offset, cov_inv, offset)
    t = (-b / (2 * a.clamp_min(1e-12))).clamp(0, 1)
    return a * t**2 + b * t + c


def compute_sparse_integral(pos, cov_inv, concentration, beams, beam_idx, gauss_idx):
    """
    compute_definite_integral restricted to the given (beam, Gaussian) pairs: the
    same closed form, evaluated pair by pair and summed per beam.

    Args:
        beam_idx, gauss_idx (Tensor (P,)): the pairs to evaluate.

    Returns:
        y_pred (Tensor (N,)): Predicted projections
    """
    origin = beams[beam_idx, 0, :]                       # (P, 2)
    v = beams[beam_idx, 1, :] - origin                   # (P, 2)
    length = torch.norm(v, dim=1)
    d = origin - pos[gauss_idx]                          # (P, 2)
    S = cov_inv[gauss_idx]                               # (P, 2, 2)

    Sv = torch.einsum('pij,pj->pi', S, v)
    A = torch.clamp(0.5 * (v * Sv).sum(-1), min=1e-8)
    B = (d * Sv).sum(-1)
    C = 0.5 * torch.einsum('pi,pij,pj->p', d, S, d)

    sqrt_A = torch.sqrt(A)
    u0 = B / (2 * sqrt_A)
    u1 = sqrt_A + u0
    integral = (math.sqrt(math.pi) / (2 * sqrt_A)) * torch.exp(B**2 / (4 * A) - C) * (torch.erf(u1) - torch.erf(u0))

    contributions = integral * length * concentration[gauss_idx]
    return torch.zeros(beams.shape[0], dtype=contributions.dtype, device=beams.device).index_add(0, beam_idx, contributions)


def estimate_concentration_scale(beams: torch.Tensor, measurements: torch.Tensor) -> float:
    """Largest path-averaged concentration of a batch, max_n y_n / l_n (ppm). 0 if no beam carries gas."""
    if measurements.numel() == 0:
        return 0.0
    lengths = torch.norm(beams[:, 1, :] - beams[:, 0, :], dim=1).clamp_min(1e-6)
    return float((measurements / lengths).max().clamp_min(0.0))
