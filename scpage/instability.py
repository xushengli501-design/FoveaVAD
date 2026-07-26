"""
SC-Page: Phase instability map computation.

Given four-phase ViT patch features [4, 14, 14, 768]:
  1. Align phases via grid_sample (offset correction)
  2. Compute per-patch cosine deviation D(p)
  3. Local MAD normalization → H(p) = ReLU((D - median) / MAD)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def compute_phase_instability(
    features: torch.Tensor,       # [4, 14, 14, 768] L2-normalized
    valid_mask: torch.Tensor,     # [4, 14, 14] bool
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute phase instability map H(p) for one frame.

    Algorithm:
      D(p) = (1/4) Σ_δ [1 - f^δ(p)^T · f̄(p)]   — mean cosine deviation
      H(p) = ReLU((D(p) - median(D)) / (MAD(D) + ε))

    Args:
        features: four-phase patch features, L2-normalized
        valid_mask: per-phase validity mask (edge ring excluded)

    Returns:
        H: [14, 14] instability map (higher = more phase-sensitive)
        D_raw: [14, 14] raw deviation map (before MAD normalization)
    """
    # Ensure same device
    valid_mask = valid_mask.to(features.device)
    n_phases, H_grid, W_grid, D = features.shape

    # ── Step 1: Align phases via sub-patch offset correction ──
    # Phase offsets (in patch units): (0,0), (0.5,0), (0,0.5), (0.5,0.5)
    # We use grid_sample to shift each phase back to the reference grid

    # Normalized coordinates for 14×14 grid
    y_coords = torch.linspace(-1, 1, H_grid, device=features.device)
    x_coords = torch.linspace(-1, 1, W_grid, device=features.device)
    gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [14, 14]

    grid = torch.stack([gx, gy], dim=-1)  # [14, 14, 2]

    phase_shifts = torch.tensor([
        [0.0, 0.0],       # (0,0) — reference
        [-0.5 / 7, 0.0],  # (8,0) — shift left by 0.5 patch in x
        [0.0, -0.5 / 7],  # (0,8) — shift up by 0.5 patch in y
        [-0.5 / 7, -0.5 / 7],  # (8,8) — shift both
    ], device=features.device)  # [4, 2]

    aligned = []
    for k in range(n_phases):
        # Build sampling grid for this phase
        g = grid + phase_shifts[k].view(1, 1, 2)  # [14, 14, 2]
        g = g.unsqueeze(0).expand(1, -1, -1, -1)  # [1, 14, 14, 2]

        # Permute features to [N, C, H, W]
        feat = features[k].permute(2, 0, 1).unsqueeze(0)  # [1, 768, 14, 14]

        aligned_feat = F.grid_sample(
            feat, g, mode='bilinear', padding_mode='border', align_corners=True,
        )  # [1, 768, 14, 14]

        aligned_feat = aligned_feat.squeeze(0).permute(1, 2, 0)  # [14, 14, 768]
        aligned_feat = F.normalize(aligned_feat, p=2, dim=-1)
        aligned.append(aligned_feat)

    aligned = torch.stack(aligned, dim=0)  # [4, 14, 14, 768]

    # ── Step 2: Compute mean feature direction f̄(p) ──
    # Only use valid phases at each position
    valid_float = valid_mask.float().unsqueeze(-1)  # [4, 14, 14, 1]
    n_valid = valid_float.sum(dim=0).clamp(min=1)  # [14, 14, 1]

    mean_feat = (aligned * valid_float).sum(dim=0) / n_valid  # [14, 14, 768]
    mean_feat = F.normalize(mean_feat, p=2, dim=-1)

    # ── Step 3: Compute D(p) = mean cosine deviation ──
    # cos_sim[k, p] = aligned[k, p] · mean_feat[p]
    cos_sim = (aligned * mean_feat.unsqueeze(0)).sum(dim=-1)  # [4, 14, 14]
    deviation = 1.0 - cos_sim  # [4, 14, 14]

    # Mask invalid phases
    deviation = deviation * valid_mask.float()
    D_raw = (deviation.sum(dim=0) / n_valid.squeeze(-1)).clamp(min=0)  # [14, 14]

    # ── Step 4: Joint valid mask across all 4 phases ──
    joint_valid = valid_mask[0].clone()
    for k in range(1, n_phases):
        joint_valid = joint_valid & valid_mask[k]

    # ── Step 5: Local MAD normalization → H(p) ──
    D_flat = D_raw[joint_valid]  # Only consider jointly valid positions
    if D_flat.numel() == 0:
        return torch.zeros(H_grid, W_grid, device=features.device), D_raw, joint_valid

    median = D_flat.median()
    mad = (D_flat - median).abs().median().clamp(min=eps)

    H = torch.relu((D_raw - median) / (mad + eps))  # [14, 14]

    # Mask invalid positions to zero
    H = H * joint_valid.float()

    return H, D_raw, joint_valid


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== Instability test ===")
    # Simulate 4-phase features: random + add a "small target" perturbation
    torch.manual_seed(42)
    features = torch.randn(4, 14, 14, 768)
    features = F.normalize(features, p=2, dim=-1)

    # Add instability at position (7, 7): make all 4 phases diverge strongly
    # This creates a region where VIT features are sensitive to phase shift
    pert_pos = (7, 7)
    for k in range(4):
        # Each phase gets a different perturbation direction at the target position
        direction = torch.randn(768)
        direction = F.normalize(direction, dim=-1)
        features[k, pert_pos[0], pert_pos[1]] = (
            0.3 * features[k, pert_pos[0], pert_pos[1]] + 0.7 * direction
        )
    features = F.normalize(features, p=2, dim=-1)

    # Valid mask: exclude outer ring
    valid = torch.ones(4, 14, 14, dtype=torch.bool)
    valid[:, 0, :] = False; valid[:, -1, :] = False
    valid[:, :, 0] = False; valid[:, :, -1] = False

    H, D = compute_phase_instability(features, valid)

    print(f"  D_raw range: [{D.min().item():.4f}, {D.max().item():.4f}]")
    print(f"  H range: [{H.min().item():.4f}, {H.max().item():.4f}]")
    print(f"  H[7,7] = {H[7,7].item():.4f}  (perturbed position, should be relatively high)")
    print(f"  H[3,3] = {H[3,3].item():.4f}  (unperturbed, should be low)")
    print(f"  H[0,0] = {H[0,0].item():.4f}  (edge, masked to 0)")
    print("=== OK! ===")
