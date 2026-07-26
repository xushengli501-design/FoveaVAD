"""
TPVA: Target-locked Progressive Visibility Audit.

Three-stage progressive context opening on the HTAG compact graph:
  M_0: C_T ← {E_P, E_T}        (past + target only)
  M_1: C_T ← {E_P, E_T, E_F1}  (add first future)
  M_2: C_T ← {E_P, E_T, E_F1, E_F2}  (add second future)

In all stages: Y ← C_T + R_static  (decision NEVER directly accesses anchors).

Outputs:
  z^0, z^1, z^2  — log-odds at each stage
  Δ^1, Δ^2       — visibility responses
  Trajectory type — 8-class classification
  Final score s_i = σ(z^2)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════
# Three-stage forward (batched)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tpva_forward(
    internvl_ll,                        # InternVLLowLevel
    global_tokens: List[torch.Tensor],   # [8] vision tokens
    local_tokens: List[torch.Tensor],    # [8] vision tokens or None
    groups: Dict[str, List[int]],        # {"P": [0,1], "T": [2,3], ...}
    audio_texts: Dict[str, Optional[str]] = None,
    prior_texts: Dict[str, Optional[str]] = None,
    active_modalities: set = None,       # {"G"} for D2, {"G","L"} for D3
    token_mode: str = None,              # 'mean'|'wmean'|'topk'|'topk_cov'
) -> Dict:
    """
    Run all three TPVA stages and return log-odds + trajectory.

    Vision tokens are encoded once via SC-Page; stage 1 HTAG encoding is
    also done once (anchor hidden states are shared); only the compact
    graph mask changes across M_0, M_1, M_2.

    Returns:
        dict with z0, z1, z2, p0, p1, p2, delta1, delta2, trajectory
    """
    from htag.graph import htag_tpva_forward as _htag_tpva

    r = _htag_tpva(
        internvl_ll, global_tokens, local_tokens, groups,
        audio_texts=audio_texts, prior_texts=prior_texts,
        active_modalities=active_modalities, token_mode=token_mode,
    )
    node_names = r['node_manifest']
    total_anchors = r['num_anchors']

    z0, z1, z2 = r['z0'], r['z1'], r['z2']
    p0, p1, p2 = r['p0'], r['p1'], r['p2']
    yes_vals = r['yes_logits']
    no_vals = r['no_logits']
    results = r['stage_details']

    # ── Visibility responses ──
    delta1 = z1 - z0  # F1 contribution
    delta2 = z2 - z1  # F2 contribution

    # ── Trajectory classification ──
    trajectory = classify_trajectory(z0, z1, z2)
    trajectory_compact = compact_trajectory(trajectory)

    return {
        'z0': z0, 'z1': z1, 'z2': z2,
        'p0': p0, 'p1': p1, 'p2': p2,
        'delta1': delta1, 'delta2': delta2,
        'trajectory': trajectory,
        'trajectory_compact': trajectory_compact,
        'final_score': p2,
        'stage_details': results,
        'yes_logits': yes_vals,
        'no_logits': no_vals,
        'node_manifest': node_names,
        'num_nodes': len(node_names),
        'num_anchors': total_anchors,
    }


# ══════════════════════════════════════════════════════════════════════
# Trajectory classification (8 types → 5 compact groups)
# ══════════════════════════════════════════════════════════════════════

def classify_trajectory(z0: float, z1: float, z2: float) -> str:
    """
    Classify the 3-stage log-odds trajectory into 8 types.

    Uses sign of log-odds (z > 0 = anomalous, z < 0 = normal).
    """
    s = lambda x: '+' if x > 0 else '-'
    t = s(z0) + ',' + s(z1) + ',' + s(z2)
    return t


def compact_trajectory(trajectory: str) -> str:
    """Map 8 types → 5 compact groups."""
    mapping = {
        '+,+,+': 'stable_anomaly',
        '-,+,+': 'delayed_confirmation',
        '-,-,+': 'delayed_confirmation',
        '+,-,-': 'context_correction',
        '+,+,-': 'context_correction',
        '+,-,+': 'oscillation',
        '-,+,-': 'oscillation',
        '-,-,-': 'stable_normal',
    }
    return mapping.get(trajectory, 'unknown')


def trajectory_to_group_idx(trajectory: str) -> int:
    """Map trajectory to group index for analysis."""
    mapping = {
        '+,+,+': 0,
        '-,+,+': 1, '-,-,+': 1,
        '+,-,-': 2, '+,+,-': 2,
        '+,-,+': 3, '-,+,-': 3,
        '-,-,-': 4,
    }
    return mapping.get(trajectory, -1)


# ══════════════════════════════════════════════════════════════════════
# Frame-level projection
# ══════════════════════════════════════════════════════════════════════

def project_to_frames(
    window_scores: List[float],       # [n_windows] s_i = p_i^2
    window_starts: List[int],          # [n_windows] start frame of window
    window_size: int = 30,
    target_start_offset: int = 8,      # offset within window where T starts
    target_end_offset: int = 14,       # offset within window where T ends
    n_frames: int = None,
) -> np.ndarray:
    """
    Project window-level scores to frame-level via triangular weighting.

    Each window writes its score only to frames in [start+8, start+14].
    Triangular weight: w(t) = 1 - |t - center| / half_width
    """
    if n_frames is None:
        n_frames = max(s + window_size for s in window_starts)

    score_sum = np.zeros(n_frames, dtype=np.float64)
    weight_sum = np.zeros(n_frames, dtype=np.float64)

    target_half = (target_end_offset - target_start_offset) / 2.0  # 3.0
    target_center = (target_start_offset + target_end_offset) / 2.0  # 11.0

    for s_i, start in zip(window_scores, window_starts):
        t_start = start + target_start_offset
        t_end = start + target_end_offset

        for t in range(t_start, t_end + 1):
            if 0 <= t < n_frames:
                w = 1.0 - abs(t - (start + target_center)) / (target_half + 1e-8)
                w = max(0.0, w)
                score_sum[t] += w * s_i
                weight_sum[t] += w

    # Normalize
    frame_scores = score_sum / np.maximum(weight_sum, 1e-8)
    return frame_scores


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== TPVA test ===")
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sage_pavad.internvl_ll import InternVLLowLevel

    wrapper = InternVLLowLevel()

    # Dummy vision tokens
    dummy_global = [torch.randn(256, 4096) for _ in range(8)]
    dummy_local = [torch.randn(256, 4096) for _ in range(8)]
    groups = {"P": [0, 1], "T": [2, 3], "F1": [4, 5], "F2": [6, 7]}

    result = tpva_forward(
        wrapper, dummy_global, dummy_local, groups,
    )

    print(f"  z^0={result['z0']:.4f}, z^1={result['z1']:.4f}, z^2={result['z2']:.4f}")
    print(f"  p^0={result['p0']:.4f}, p^1={result['p1']:.4f}, p^2={result['p2']:.4f}")
    print(f"  Δ^1={result['delta1']:.4f}, Δ^2={result['delta2']:.4f}")
    print(f"  trajectory: {result['trajectory']} → {result['trajectory_compact']}")
    print(f"  final_score: {result['final_score']:.4f}")

    # Test trajectory classification
    print("\n  Trajectory tests:")
    test_cases = [
        (1.0, 2.0, 3.0),   # +,+,+
        (-1.0, 1.0, 2.0),  # -,+,+
        (-1.0, -1.0, 1.0), # -,-,+
        (1.0, -1.0, -1.0), # +,-,-
        (-1.0, -1.0, -1.0),# -,-,-
        (1.0, -1.0, 1.0),  # +,-,+
    ]
    for z0, z1, z2 in test_cases:
        t = classify_trajectory(z0, z1, z2)
        tc = compact_trajectory(t)
        print(f"    ({z0:+.0f}, {z1:+.0f}, {z2:+.0f}) → {t} → {tc}")

    # Test frame projection
    scores = [0.8, 0.9, 0.3]
    starts = [0, 4, 8]
    fs = project_to_frames(scores, starts, n_frames=30)
    print(f"\n  Frame scores: shape={fs.shape}, range=[{fs.min():.3f}, {fs.max():.3f}]")

    print("=== OK! ===")
