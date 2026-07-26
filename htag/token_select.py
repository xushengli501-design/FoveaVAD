"""
Spatial token selection for HTAG evidence nodes.

Replaces destructive mean-pool (256 tokens → 1 vector) with:
  - SST Top-K salient patches (motion-aware)
  - Uniform coverage patches (scene context)
  - 4×4 adaptive spatial pooling (for local crops)

Key: preserves per-frame spatial structure (no temporal mean-pool).
     Early and late frame tokens are kept SEPARATE.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


# ══════════════════════════════════════════════════════════════════════
# SST patch weights (simplified: spatial + temporal gradient energy)
# ══════════════════════════════════════════════════════════════════════

def compute_simple_sst_weights(
    frame1_tokens: torch.Tensor,  # [256, D]
    frame2_tokens: torch.Tensor,  # [256, D]
) -> torch.Tensor:
    """
    Compute per-patch saliency weights from frame-to-frame change + spatial variance.

    w(p) = ||v2(p) - v1(p)||₂ + λ · Var_spatial(v2(p))

    Higher weight → more likely to contain motion or texture change.
    Returns [256] normalized weights (sum to 1).
    """
    # Temporal change
    temporal_diff = (frame2_tokens - frame1_tokens).norm(dim=-1)  # [256]

    # Spatial variance (within 3×3 neighborhood of frame 2)
    f2 = frame2_tokens.reshape(16, 16, -1)  # [16, 16, D]
    f2_padded = F.pad(f2.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    f2_padded = f2_padded.squeeze(0).permute(1, 2, 0)  # [18, 18, D]

    spatial_var = torch.zeros(256, device=frame1_tokens.device)
    for i in range(16):
        for j in range(16):
            patch = f2_padded[i:i + 3, j:j + 3, :].reshape(9, -1)  # [9, D]
            spatial_var[i * 16 + j] = patch.std(dim=0).norm()

    lam = 0.3
    weights = temporal_diff + lam * spatial_var
    weights = weights / (weights.sum() + 1e-12)
    return weights


# ══════════════════════════════════════════════════════════════════════
# Uniform coverage grid
# ══════════════════════════════════════════════════════════════════════

def get_uniform_grid_indices(
    grid_h: int = 16,
    grid_w: int = 16,
    num_points: int = 8,
) -> torch.Tensor:
    """
    Select num_points uniformly distributed positions on a grid_h × grid_w grid.

    Uses a strided sampling pattern to ensure coverage.
    Returns [num_points] linear indices.
    """
    if num_points <= 0:
        return torch.tensor([], dtype=torch.long)

    total = grid_h * grid_w

    # Determine stride to cover the grid
    stride_h = max(1, grid_h // int(np.ceil(np.sqrt(num_points * grid_h / grid_w))))
    stride_w = max(1, grid_w // int(np.ceil(np.sqrt(num_points * grid_w / grid_h))))

    indices = []
    for i in range(0, grid_h, stride_h):
        for j in range(0, grid_w, stride_w):
            indices.append(i * grid_w + j)
            if len(indices) >= num_points:
                break
        if len(indices) >= num_points:
            break

    # If not enough, fill remaining
    if len(indices) < num_points:
        step = total // (num_points - len(indices) + 1)
        for i in range(step, total, step):
            if i not in indices:
                indices.append(i)
                if len(indices) >= num_points:
                    break

    return torch.tensor(indices[:num_points], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════
# Global token selection
# ══════════════════════════════════════════════════════════════════════

def select_global_tokens(
    frame1_tokens: torch.Tensor,    # [256, D]
    frame2_tokens: torch.Tensor,    # [256, D]
    sst_weights: torch.Tensor,      # [256]
    k_salient: int = 8,
    k_coverage: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select K = k_salient + k_coverage patches per frame.

    Returns:
        early_tokens: [K, D] — selected tokens from frame 1
        late_tokens: [K, D] — selected tokens from frame 2 (same positions)
        selected_idx: [K] — linear indices (0..255)
    """
    # Salient: Top-K by SST weight
    salient_idx = torch.topk(sst_weights, k=k_salient).indices

    # Coverage: uniform grid
    coverage_idx = get_uniform_grid_indices(16, 16, k_coverage).to(sst_weights.device)

    # Merge, deduplicate, preserve order
    all_idx = torch.cat([salient_idx, coverage_idx])
    # Deduplicate while preserving order
    seen = set()
    unique_idx = []
    for idx in all_idx.tolist():
        if idx not in seen:
            seen.add(idx)
            unique_idx.append(idx)
    selected_idx = torch.tensor(unique_idx[:k_salient + k_coverage], device=sst_weights.device)

    early_tokens = frame1_tokens[selected_idx]  # [K, D]
    late_tokens = frame2_tokens[selected_idx]   # [K, D]

    return early_tokens, late_tokens, selected_idx


# ══════════════════════════════════════════════════════════════════════
# Local token selection (4×4 spatial pooling)
# ══════════════════════════════════════════════════════════════════════

def spatial_pool_4x4(tokens: torch.Tensor) -> torch.Tensor:
    """
    Adaptive 4×4 spatial pooling: [256, D] → [16, D].

    Preserves spatial layout while reducing dimensionality.
    Works even when tokens don't form a perfect 16×16 grid.
    """
    D = tokens.shape[-1]
    # Try to reshape to square grid
    grid_size = int(np.sqrt(tokens.shape[0]))
    if grid_size * grid_size == tokens.shape[0]:
        x = tokens.reshape(grid_size, grid_size, D)
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = x.squeeze(0).permute(1, 2, 0)  # [4, 4, D]
        return x.reshape(16, D)
    else:
        # Non-square: use 1D adaptive pooling
        x = tokens.unsqueeze(0).permute(0, 2, 1)  # [1, D, N]
        x = F.adaptive_avg_pool1d(x, 16)
        x = x.squeeze(0).permute(1, 0)  # [16, D]
        return x


def select_local_tokens(
    frame1_tokens: torch.Tensor,    # [N, D] (may be non-square)
    frame2_tokens: torch.Tensor,    # [N, D]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select local tokens via 4×4 spatial pooling (no SST dependence).

    Returns:
        early_tokens: [16, D]
        late_tokens: [16, D]
    """
    early = spatial_pool_4x4(frame1_tokens)
    late = spatial_pool_4x4(frame2_tokens)
    return early, late


# ══════════════════════════════════════════════════════════════════════
# Weighted mean (P1 ablation)
# ══════════════════════════════════════════════════════════════════════

def sst_weighted_mean(
    frame1_tokens: torch.Tensor,    # [N, D]
    frame2_tokens: torch.Tensor,    # [N, D]
    sst_weights: torch.Tensor,      # [N]
) -> torch.Tensor:
    """
    SST-weighted mean over both frames: 256→1 vector.

    v̄ = Σ_p w_p · (v1_p + v2_p) / (2 · Σ w_p)
    """
    w = sst_weights / (sst_weights.sum() + 1e-12)
    avg = (frame1_tokens + frame2_tokens) / 2.0
    return (w.unsqueeze(-1) * avg).sum(dim=0)  # [D]


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== Token selection test ===")
    torch.manual_seed(42)

    f1 = torch.randn(256, 4096)
    f2 = torch.randn(256, 4096)

    # SST weights
    w = compute_simple_sst_weights(f1, f2)
    print(f"SST weights: shape={w.shape}, range=[{w.min():.6f}, {w.max():.6f}]")

    # P0: mean
    p0 = 0.5 * (f1.mean(dim=0) + f2.mean(dim=0))
    print(f"P0 (mean): shape={p0.shape}")

    # P1: SST weighted mean
    p1 = sst_weighted_mean(f1, f2, w)
    print(f"P1 (SST w-mean): shape={p1.shape}")

    # P2: SST Top-16
    e2, l2, idx2 = select_global_tokens(f1, f2, w, k_salient=16, k_coverage=0)
    print(f"P2 (Top-16): early={e2.shape}, late={l2.shape}")

    # P3: SST Top-8 + uniform 8
    e3, l3, idx3 = select_global_tokens(f1, f2, w, k_salient=8, k_coverage=8)
    print(f"P3 (Top-8+cov-8): early={e3.shape}, late={l3.shape}, idx={idx3}")

    # Local: 4×4 pool
    fl1 = torch.randn(256, 4096)
    fl2 = torch.randn(256, 4096)
    le, ll = select_local_tokens(fl1, fl2)
    print(f"Local 4×4: early={le.shape}, late={ll.shape}")

    print("=== OK! ===")
