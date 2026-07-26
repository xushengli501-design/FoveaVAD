"""
SAGE-PAVAD data pipeline: Video reading → frame preprocessing → window extraction.

Connects to UCF-Crime videos using decord + InternVL preprocessing.
"""

import torch
import numpy as np
import os, sys
from typing import List, Tuple, Optional
from pathlib import Path

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, '/sdb/data_public/llms/llm/InternVL2-8B')
sys.path.insert(0, os.path.join(_PROJ, 'CoReVAD', 'utils'))  # for internvl_utils

from PIL import Image
from decord import VideoReader, cpu as dcpu
from internvl_utils import build_transform, dynamic_preprocess


# ══════════════════════════════════════════════════════════════════════
# Video reading
# ══════════════════════════════════════════════════════════════════════

def load_video_frames(
    video_path: str,
    max_frames: int = None,
) -> Tuple[List[np.ndarray], int]:
    """
    Load all frames from a video file.

    Returns:
        frames: list of [H, W, 3] numpy arrays
        total_frames: int
    """
    vr = VideoReader(video_path, ctx=dcpu(0), num_threads=1)
    total = len(vr)

    n_to_load = total if max_frames is None else min(total, max_frames)
    frames = []
    for i in range(n_to_load):
        frames.append(vr[i].asnumpy())

    return frames, total


def load_video_segment(
    video_path: str,
    start_frame: int,
    window_size: int = 30,
) -> List[np.ndarray]:
    """Load a contiguous segment of 30 frames from a video."""
    vr = VideoReader(video_path, ctx=dcpu(0), num_threads=1)
    total = len(vr)

    end = min(start_frame + window_size, total)
    frames = []
    for i in range(start_frame, end):
        frames.append(vr[i].asnumpy())

    # Pad if needed
    while len(frames) < window_size:
        frames.append(frames[-1].copy())

    return frames


# ══════════════════════════════════════════════════════════════════════
# Image preprocessing for InternVL
# ══════════════════════════════════════════════════════════════════════

def preprocess_for_internvl(
    frame: np.ndarray,       # [H, W, 3] uint8
    image_size: int = 448,
    max_num_tiles: int = 1,
) -> torch.Tensor:
    """
    Preprocess a single frame for InternVL2-8B.

    Returns:
        pixel_values: [1, 3, 448, 448] tensor
    """
    img = Image.fromarray(frame).convert('RGB')
    img = dynamic_preprocess(img, image_size=image_size, use_thumbnail=True, max_num=max_num_tiles)

    transform = build_transform(input_size=image_size)
    pv = torch.stack([transform(t) for t in img])  # [num_tiles, 3, 448, 448]
    return pv


def preprocess_for_vit(
    frame: np.ndarray,       # [H, W, 3] uint8
    image_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess a single frame for ViT-B/16 (SC-Page).

    Returns:
        tensor: [3, 224, 224] in [0, 1]
    """
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

    # Standard ViT preprocessing
    transform = Compose([
        Resize(image_size, interpolation=3),  # bicubic
        CenterCrop(image_size),
        ToTensor(),  # → [0, 1]
    ])

    img = Image.fromarray(frame).convert('RGB')
    return transform(img)


# ══════════════════════════════════════════════════════════════════════
# Diagnostic window extraction
# ══════════════════════════════════════════════════════════════════════

def extract_diagnostic_windows(
    video_path: str,
    gt_intervals: List[Tuple[int, int]],  # [(start, end), ...] ground-truth anomaly intervals
    window_size: int = 30,
    stride: int = 4,
) -> List[dict]:
    """
    Extract 3 diagnostic windows from a video:
      1. Normal window (target T has no anomaly, future has no anomaly)
      2. Anomalous window (target T contains anomaly)
      3. Future-leak window (target T is normal, but F1/F2 contains anomaly)

    Returns list of dicts with metadata for each window.
    """
    vr = VideoReader(video_path, ctx=dcpu(0), num_threads=1)
    total = len(vr)

    def is_anomalous(start, end):
        """Check if frame interval [start, end) overlaps any GT anomaly."""
        for gt_s, gt_e in gt_intervals:
            if start < gt_e and end > gt_s:
                return True
        return False

    # Window T interval = [start+8, start+14]
    # Window F1 interval = [start+15, start+22]
    # Window F2 interval = [start+23, start+29]

    windows = []

    # Generate all possible windows
    for start in range(0, total - window_size, stride):
        t_start, t_end = start + 8, start + 14
        f1_start, f1_end = start + 15, start + 22
        f2_start, f2_end = start + 23, start + 29

        t_anom = is_anomalous(t_start, t_end)
        f1_anom = is_anomalous(f1_start, f1_end)
        f2_anom = is_anomalous(f2_start, f2_end)

        # Type 1: T normal, future normal
        if not t_anom and not f1_anom and not f2_anom:
            if not any(w['type'] == 'normal' for w in windows):
                windows.append({
                    'type': 'normal',
                    'start': start,
                    't_anom': False, 'f1_anom': False, 'f2_anom': False,
                })

        # Type 2: T anomalous
        if t_anom:
            if not any(w['type'] == 'target_anomaly' for w in windows):
                windows.append({
                    'type': 'target_anomaly',
                    'start': start,
                    't_anom': True, 'f1_anom': f1_anom, 'f2_anom': f2_anom,
                })

        # Type 3: T normal, but future has anomaly (future leak risk)
        if not t_anom and (f1_anom or f2_anom):
            if not any(w['type'] == 'future_leak' for w in windows):
                windows.append({
                    'type': 'future_leak',
                    'start': start,
                    't_anom': False, 'f1_anom': f1_anom, 'f2_anom': f2_anom,
                })

        if len(windows) >= 3:
            break

    return windows


# ══════════════════════════════════════════════════════════════════════
# UCF-Crime dataset helpers
# ══════════════════════════════════════════════════════════════════════

UCF_VIDEO_DIR = '/sdb/data_public/llms/videos/UCFcrime/videos'
UCF_TEST_LIST = '/sdb/data_public/llms/videos/UCFcrime/Anomaly_Detection_splits/Anomaly_Test.txt'
UCF_GT_DIR = '/sdb/data_public/llms/videos/UCFcrime'


def load_ucf_gt(video_name: str) -> List[Tuple[int, int]]:
    """
    Load ground-truth anomaly intervals for a UCF-Crime video.

    GT format: one file per video, each line = "start_frame end_frame"
    """
    import glob

    # Find the GT file
    patterns = [
        os.path.join(UCF_GT_DIR, '**', f'{video_name}*.txt'),
        os.path.join(UCF_GT_DIR, '**', '*.txt'),
    ]
    # UCF GT files are named by category/video
    # The anomaly intervals are stored per video
    gt_path = None
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        for m in matches:
            if video_name in os.path.basename(m) or video_name in m:
                gt_path = m
                break
        if gt_path:
            break

    if gt_path is None:
        # Try the cached GT from CoReVAD
        cached_gt = '/sda/home/temp/lixusheng/HyperVAD/CoReVAD/src/ucf/gt_ucf.npy'
        if os.path.exists(cached_gt):
            # This is a frame-level GT array, not intervals
            return None  # Will handle differently

    if gt_path and os.path.exists(gt_path):
        intervals = []
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        intervals.append((int(parts[0]), int(parts[1])))
        return intervals

    return None


def find_video_path(video_name: str) -> Optional[str]:
    """Find a UCF video file by name."""
    import glob
    # Direct match
    direct = os.path.join(UCF_VIDEO_DIR, f'{video_name}.mp4')
    if os.path.exists(direct):
        return direct

    # Search in subdirectories
    pattern = os.path.join(UCF_VIDEO_DIR, '**', f'{video_name}.mp4')
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    return None


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== Data pipeline test ===")

    # Find a UCF test video
    with open(UCF_TEST_LIST) as f:
        test_videos = [l.strip().replace('.mp4', '').split('/')[-1] for l in f if l.strip()]

    print(f"  UCF test videos: {len(test_videos)}")

    # Try to load the first video
    for vname in test_videos[:5]:
        vpath = find_video_path(vname)
        if vpath:
            print(f"  Loading: {vname} → {vpath}")
            try:
                vr = VideoReader(vpath, ctx=dcpu(0), num_threads=1)
                n_frames = len(vr)
                print(f"    Frames: {n_frames}, fps: {vr.get_avg_fps():.1f}")

                # Load first frame
                frame = vr[0].asnumpy()
                print(f"    Frame shape: {frame.shape}")

                # Test preprocessing
                pv = preprocess_for_internvl(frame)
                print(f"    InternVL pixel_values: {pv.shape}")

                vit_in = preprocess_for_vit(frame)
                print(f"    ViT input: {vit_in.shape}")

                break
            except Exception as e:
                print(f"    Error: {e}")
