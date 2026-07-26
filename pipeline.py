"""
SAGE-PAVAD full pipeline.

SC-Page (ViT phase instability → local evidence crops)
  → HTAG (typed evidence graph with C_T audit)
    → TPVA (three-stage progressive visibility)
      → Frame projection & post-processing.

All models frozen. No training. Single frame-level anomaly score.
"""

import torch
import numpy as np
import os, sys, time
from typing import Dict, List, Tuple, Optional

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from internvl_ll import InternVLLowLevel
from scpage import SCPageExtractor
from tpva import tpva_forward, project_to_frames


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

class SAGEConfig:
    """Unified SAGE-PAVAD configuration."""
    # ── Window ──
    window_size: int = 30
    sample_count: int = 8
    window_stride: int = 4
    sample_offsets = [0, 4, 8, 12, 17, 21, 25, 29]

    # ── Temporal groups ──
    temporal_groups = {"P": [0, 1], "T": [2, 3], "F1": [4, 5], "F2": [6, 7]}
    target_start = 8
    target_end = 14
    target_center = 11

    # ── SC-Page ──
    vit_path: str = '/sdb/data_public/llms/vit-base-patch16-224-in21k'
    scpage_box_size: int = 7
    scpage_expand_ratio: float = 1.10
    phase_cache_size: int = 256

    # ── HTAG ──
    num_anchors_per_node: int = 4
    llm_stage1_end: int = 23

    # ── TPVA ──
    # (no tunable parameters — just reads masks)

    # ── Post-processing ──
    gaussian_size: int = 3
    gaussian_sigma: float = 2.0
    position_prior: float = 0.5
    position_center: float = 0.5

    # ── Paths ──
    internvl_path: str = '/sdb/data_public/llms/llm/InternVL2-8B'


# ══════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════

class SAGEPipeline:
    """SAGE-PAVAD full pipeline for one video."""

    def __init__(self, cfg: SAGEConfig = None):
        self.cfg = cfg or SAGEConfig()
        self.internvl = None     # lazy init
        self.scpage = None       # lazy init

    def _ensure_models(self):
        if self.internvl is None:
            self.internvl = InternVLLowLevel(self.cfg.internvl_path)
        if self.scpage is None:
            self.scpage = SCPageExtractor(
                vit_path=self.cfg.vit_path,
                device=self.internvl.device,
                box_size=self.cfg.scpage_box_size,
                expand_ratio=self.cfg.scpage_expand_ratio,
                cache_size=self.cfg.phase_cache_size,
            )

    @torch.no_grad()
    def process_window(
        self,
        window_frames: List[np.ndarray],   # [30] raw frames (H, W, 3) numpy arrays
        window_start: int,                  # global start frame index
        audio_segment: Optional[Dict] = None,  # optional audio for this window
        prior_segment: Optional[Dict] = None,  # optional PAVAD prior per time block
    ) -> Dict:
        """
        Process one 30-frame window through SC-Page → HTAG → TPVA.

        Returns window-level dict with z^0..z^2, p^2, trajectory, etc.
        """
        self._ensure_models()
        cfg = self.cfg

        # ── 1. Sample 8 frames ──
        sampled = [window_frames[i] for i in cfg.sample_offsets]
        frame_ids = [window_start + i for i in cfg.sample_offsets]

        # Convert to tensors [3, H, W]
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        sampled_tensors = [to_tensor(f) for f in sampled]

        # ── 2. SC-Page: global + local images ──
        global_imgs, local_crops = self.scpage.process_window(
            sampled_tensors, frame_ids,
        )
        # global_imgs[i]: [1, 3, 448, 448], local_crops[i]: [1, 3, 448, 448]

        # Batch encode all images through InternVL vision encoder
        all_imgs = torch.cat(global_imgs + local_crops, dim=0)  # [16, 3, 448, 448]
        vis_tokens = self.internvl.encode_images(all_imgs)  # [16, 256, 4096]

        global_tokens = list(vis_tokens[:8])   # [8] each [256, 4096]
        local_tokens = list(vis_tokens[8:])    # [8] each [256, 4096]

        # ── 3. PAVAD prior (time-partitioned) ──
        prior_texts = {}
        if prior_segment is not None:
            for g in ['P', 'T', 'F1', 'F2']:
                if g in prior_segment and prior_segment[g]:
                    prior_texts[g] = prior_segment[g]

        # ── 4. Audio (optional) ──
        audio_texts = {}
        if audio_segment is not None:
            for g in ['P', 'T', 'F1', 'F2']:
                if g in audio_segment and audio_segment[g]:
                    audio_texts[g] = audio_segment[g]

        # ── 5. HTAG + TPVA ──
        active_mods = {"G", "L"}  # default: global + local
        result = tpva_forward(
            self.internvl,
            global_tokens, local_tokens,
            cfg.temporal_groups,
            audio_texts=audio_texts if audio_texts else None,
            prior_texts=prior_texts if prior_texts else None,
            active_modalities=active_mods,
        )

        return result

    def process_video(
        self,
        frames: List[np.ndarray],           # [N] raw frames
        audio: Optional[List] = None,       # optional audio per frame
        prior_fn=None,                      # function(frame_list, start, end) → prior texts
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Process full video → frame-level anomaly scores.

        Args:
            frames: list of [H, W, 3] numpy arrays
            audio: optional per-frame audio data
            prior_fn: optional function to compute PAVAD prior per time block
            verbose: print progress

        Returns:
            frame_scores: [N] anomaly scores in [0, 1]
        """
        self._ensure_models()
        cfg = self.cfg
        N = len(frames)

        if N < cfg.window_size:
            # Pad by repeating boundary frames
            n_pad = cfg.window_size - N
            frames = [frames[0]] * (n_pad // 2) + list(frames) + [frames[-1]] * (n_pad - n_pad // 2)
            N = len(frames)

        # Generate windows
        window_scores = []
        window_starts = []
        trajectories = []
        z_sequences = []

        n_windows = max(0, (N - cfg.window_size) // cfg.window_stride + 1)
        if verbose:
            print(f"[SAGE-PAVAD] {N} frames → {n_windows} windows", flush=True)

        t_start = time.time()
        for wi in range(n_windows):
            start = wi * cfg.window_stride
            end = start + cfg.window_size
            window_frames = frames[start:end]

            # Compute PAVAD prior if function provided
            prior = None
            if prior_fn is not None:
                prior = {}
                for g, idxs in cfg.temporal_groups.items():
                    g_start = start + cfg.sample_offsets[idxs[0]]
                    g_end = start + cfg.sample_offsets[idxs[-1]] + 1
                    prior[g] = prior_fn(window_frames, g_start, g_end)

            # Process window
            result = self.process_window(
                window_frames, start,
                prior_segment=prior,
            )

            window_scores.append(result['final_score'])
            window_starts.append(start)
            trajectories.append(result['trajectory_compact'])
            z_sequences.append((result['z0'], result['z1'], result['z2']))

            if verbose and (wi + 1) % 10 == 0:
                elapsed = time.time() - t_start
                fps = (wi + 1) * cfg.window_stride / elapsed
                print(f"  [{wi+1}/{n_windows}] {fps:.1f} fps", flush=True)

        if verbose:
            elapsed = time.time() - t_start
            print(f"  Done: {n_windows} windows in {elapsed:.1f}s ({N/elapsed:.1f} fps)", flush=True)

        # ── Frame-level projection ──
        frame_scores = project_to_frames(
            window_scores, window_starts,
            window_size=cfg.window_size,
            n_frames=N,
        )

        # ── Post-processing: Gaussian smooth + position prior ──
        frame_scores = self._postproc(frame_scores)

        # ── Clip to original length (remove padding) ──
        # (actual video length should be tracked above; for now, return all)

        return frame_scores, {
            'trajectories': trajectories,
            'z_sequences': z_sequences,
        }

    def _postproc(self, scores: np.ndarray) -> np.ndarray:
        """Gaussian smoothing + position prior."""
        cfg = self.cfg
        # Gaussian smooth
        if cfg.gaussian_size > 1 and len(scores) >= 3:
            try:
                from scipy.ndimage import gaussian_filter1d
                scores = gaussian_filter1d(scores, sigma=cfg.gaussian_sigma)
            except Exception:
                pass

        # Position prior
        if cfg.position_prior > 0:
            n = len(scores)
            t = np.arange(n, dtype=np.float64)
            ctr = n * cfg.position_center
            sigma_pos = n * cfg.position_prior + 1e-9
            prior = np.exp(-0.5 * ((t - ctr) / sigma_pos) ** 2)
            scores = scores * prior

        # Min-max normalize
        mn, mx = scores.min(), scores.max()
        if mx - mn > 1e-9:
            scores = (scores - mn) / (mx - mn)

        return scores


# ══════════════════════════════════════════════════════════════════════
# Quick test (dummy video)
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== SAGE-PAVAD pipeline test ===")

    pipe = SAGEPipeline()
    pipe._ensure_models()

    # Create a tiny dummy video (60 frames, 224×224×3)
    n_frames = 60
    dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    for _ in range(n_frames)]

    scores, meta = pipe.process_video(dummy_frames, verbose=False)
    print(f"  Frame scores: shape={scores.shape}")
    print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Trajectories: {len(meta['trajectories'])} windows")

    # Count trajectory types
    from collections import Counter
    tc = Counter(meta['trajectories'])
    print(f"  Trajectory distribution: {dict(tc)}")

    print("=== OK! ===")
