import torch

from config import Config


def injection_scale(cfg: Config) -> float:
    """Standard deviation of injected Gaussians, also the minimum spacing between them (m)."""
    if cfg.densify.injection_scale is not None:
        return cfg.densify.injection_scale
    return min(cfg.env.map_size) * 0.1


def extract_candidate_positions(
    beams: torch.Tensor,
    importance_scores: torch.Tensor,
    min_dist: float,
    device: torch.device,
    predictions: torch.Tensor | None = None,
    existing_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Extracts locations for new Gaussians by intersecting the beams the model
    does not explain (or taking the midpoint of those that cross no other one),
    and thinning the result by spatial non-maximum suppression.

    Args:
        beams: Tensor of shape (N, 2, 2) representing the start and end points of the beams.
        importance_scores: Tensor of shape (N,) with the gas missing along each beam,
            i.e. the residual y - y_hat. On an empty model this is simply y. It first
            decides which beams are admitted, and then ranks the admitted ones.
        min_dist: Minimum spatial distance between generated points to avoid redundancy.
        device: Torch device.
        predictions: Optional tensor of shape (N,) with the current prediction y_hat of
            each beam. A beam is admitted only when it misses more gas than the model
            accounts for, y - y_hat > y_hat, so that injection responds to regions the
            optimizer cannot reach rather than to whichever beams happen to fit worst.
            Pass None (e.g. during initialization) to use the empty-model prediction
            y_hat = 0, which admits every beam carrying gas.
        existing_pos: Optional tensor of shape (K, 2) with positions of already-placed Gaussians.
            Candidates within min_dist of any of these are discarded before NMS. Pass None
            (e.g. during initialization, when no Gaussians exist yet) to skip this filter.

    Returns:
        Tensor of shape (M, 2) with the filtered candidate positions.
    """
    empty = torch.empty((0, 2), device=device)

    if predictions is None:
        predictions = torch.zeros_like(importance_scores)

    # 1. Admissibility: the beam misses more gas than the model explains. Where the
    # model does predict gas, primitives already lie along the beam and the optimizer
    # can reduce the residual on its own, so no injection is warranted. On an empty
    # model the prediction is zero and this reduces to "the beam carries gas".
    # Both sides share the length of the beam, so the test needs no normalization.
    admitted = importance_scores > predictions
    num_admitted = int(admitted.sum())
    if num_admitted == 0:
        return empty

    # 2. The admitted beams are scored by the average concentration missing along
    # them, dividing by the length so that long and short beams are compared on the
    # same footing, and those reaching the mean plus one standard deviation are kept.
    # The spread adapts the selectivity on its own. A wide spread means some beams
    # genuinely stand out and only those are kept, while a narrow spread means every
    # admitted beam reports the same missing gas and all of them are kept, which is
    # why the comparison is not strict.
    lengths = torch.clamp(torch.norm(beams[:, 1, :] - beams[:, 0, :], dim=1), min=1e-5)
    normalized_scores = importance_scores / lengths

    admitted_scores = normalized_scores[admitted]
    spread = torch.std(admitted_scores) if num_admitted > 1 else 0.0
    threshold = torch.mean(admitted_scores) + spread

    candidate_mask = admitted & (normalized_scores >= threshold)
    candidate_indices = torch.where(candidate_mask)[0]

    num_candidates = candidate_indices.shape[0]
    if num_candidates == 0:
        return empty

    # 3. Geometric Intersections of candidate segments
    candidate_beams = beams[candidate_indices]
    candidate_scores = normalized_scores[candidate_indices]
    
    i, j = torch.triu_indices(num_candidates, num_candidates, offset=1, device=device)
    
    b_i = candidate_beams[i] 
    b_j = candidate_beams[j] 
    
    x1, y1 = b_i[:, 0, 0], b_i[:, 0, 1]
    x2, y2 = b_i[:, 1, 0], b_i[:, 1, 1]
    x3, y3 = b_j[:, 0, 0], b_j[:, 0, 1]
    x4, y4 = b_j[:, 1, 0], b_j[:, 1, 1]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    valid_intersection = torch.abs(denom) > 1e-7
    
    # Parametric line equations (t for beam_i, u for beam_j)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / (denom + 1e-9)
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / (denom + 1e-9)
    
    # A true segment intersection ONLY occurs if both t and u are between 0 and 1.
    # We use 0.01 and 0.99 to avoid intersections exactly at the emitters (radial origins).
    in_bounds = (t >= 0.01) & (t <= 0.99) & (u >= 0.01) & (u <= 0.99)
    keep_mask = valid_intersection & in_bounds
        
    # Calculate exact coordinates for valid intersections
    px = x1[keep_mask] + t[keep_mask] * (x2[keep_mask] - x1[keep_mask])
    py = y1[keep_mask] + t[keep_mask] * (y2[keep_mask] - y1[keep_mask])
    
    intersections = torch.stack([px, py], dim=-1)
    int_scores = candidate_scores[i[keep_mask]] + candidate_scores[j[keep_mask]]

    # A candidate that crosses no other candidate still reports gas missing somewhere
    # along it, so its midpoint stands in for the unknown position. It is scored by its
    # own residual alone, below any intersection, which NMS therefore prefers
    crossed = torch.zeros(num_candidates, dtype=torch.bool, device=device)
    crossed[i[keep_mask]] = True
    crossed[j[keep_mask]] = True
    midpoints = candidate_beams[~crossed].mean(dim=1)
    intersections = torch.cat([intersections, midpoints], dim=0)
    int_scores = torch.cat([int_scores, candidate_scores[~crossed]], dim=0)

    # 4. Discard candidates too close to already-placed Gaussians
    if existing_pos is not None and existing_pos.shape[0] > 0:
        dist_to_existing = torch.cdist(intersections, existing_pos)
        novel_mask = torch.min(dist_to_existing, dim=1).values >= min_dist
        if not novel_mask.any():
            return empty
        intersections = intersections[novel_mask]
        int_scores = int_scores[novel_mask]

    # 5. Continuous Spatial Non-Maximum Suppression (NMS).
    # Greedy suppression, iterating over the points that are kept rather than over the
    # candidates: at every pass the highest-scoring survivor is accepted and every
    # candidate within min_dist of it is discarded in a single vectorized operation.
    # The number of passes is therefore bounded by how many primitives fit in the
    # domain at that spacing, not by the number of candidates.
    order = torch.argsort(int_scores, descending=True)
    points = intersections[order]

    alive = torch.ones(points.shape[0], dtype=torch.bool, device=device)
    keep_indices = []
    while bool(alive.any()):
        # Points are sorted by score, so the first survivor is the highest-scoring one
        idx = torch.nonzero(alive, as_tuple=False)[0]
        keep_indices.append(idx)
        # Suppresses the accepted point as well, since its own distance to itself is 0
        alive &= torch.norm(points - points[idx], dim=1) >= min_dist

    return points[torch.cat(keep_indices)]
