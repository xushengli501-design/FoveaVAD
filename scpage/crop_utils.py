"""
SC-Page: Box search + causal smoothing + coordinate mapping + cropping.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


# ══════════════════════════════════════════════════════════════════════
# Box search on instability heatmap
# ══════════════════════════════════════════════════════════════════════

def compute_padding_mask(
    orig_w: int, orig_h: int,
    vit_size: int = 224,
    patch_size: int = 16,
    grid_size: int = 14,
) -> torch.Tensor:
    """
    Compute which patch grid positions are in letterbox padding.

    For a (orig_w × orig_h) image resized to vit_size×vit_size with padding:
      scale = min(vit_size / orig_w, vit_size / orig_h)
      new_w, new_h = int(orig_w * scale), int(orig_h * scale)
      pad_x = (vit_size - new_w) // 2, pad_y = (vit_size - new_h) // 2

    Returns [grid_size, grid_size] bool mask (True = valid content, False = padding).
    """
    scale = min(vit_size / orig_w, vit_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (vit_size - new_w) // 2
    pad_y = (vit_size - new_h) // 2

    content_left = pad_x
    content_right = pad_x + new_w
    content_top = pad_y
    content_bottom = pad_y + new_h

    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    for r in range(grid_size):
        for c in range(grid_size):
            patch_left = c * patch_size
            patch_right = (c + 1) * patch_size
            patch_top = r * patch_size
            patch_bottom = (r + 1) * patch_size

            # Patch must be FULLY inside content area
            if (patch_left >= content_left and patch_right <= content_right and
                patch_top >= content_top and patch_bottom <= content_bottom):
                mask[r, c] = True

    # Erode by 1 patch to avoid boundary artifacts
    eroded = mask.clone()
    for r in range(1, grid_size - 1):
        for c in range(1, grid_size - 1):
            if not mask[r - 1:r + 2, c - 1:c + 2].all():
                eroded[r, c] = False
    eroded[0, :] = False; eroded[-1, :] = False
    eroded[:, 0] = False; eroded[:, -1] = False

    return eroded


def search_max_energy_box(
    H: torch.Tensor,               # [14, 14] instability map
    box_size: int = 5,              # default 5×5 (was 7×7)
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """
    Find the box_size×box_size region with maximum mean instability energy.

    Candidate must be 100% inside valid_mask (no partial overlap).
    Uses integral image for O(1) per candidate.

    Returns:
        (top_left_row, top_left_col) or None if no valid candidate, energy
    """
    H_grid = H.shape[0]
    if valid_mask is None:
        valid_mask = torch.ones(H_grid, H_grid, dtype=torch.bool, device=H.device)

    # Build integral image
    integral = torch.zeros(H_grid + 1, H_grid + 1, device=H.device, dtype=H.dtype)
    integral[1:, 1:] = H.cumsum(dim=0).cumsum(dim=1)

    best_energy = float('-inf')
    best_pos = None

    max_start = H_grid - box_size
    for i in range(max_start + 1):
        for j in range(max_start + 1):
            # Require 100% valid (no padding overlap)
            if not valid_mask[i:i + box_size, j:j + box_size].all():
                continue

            area_sum = (
                integral[i + box_size, j + box_size]
                - integral[i, j + box_size]
                - integral[i + box_size, j]
                + integral[i, j]
            )
            energy = area_sum / (box_size * box_size)

            if energy > best_energy:
                best_energy = energy
                best_pos = (i, j)

    return best_pos, best_energy


# ══════════════════════════════════════════════════════════════════════
# Causal median smoothing of box centers
# ══════════════════════════════════════════════════════════════════════

def causal_median_center(
    centers: List[Tuple[float, float]],  # [(cx, cy), ...] history including current
    current_index: int,
) -> Tuple[float, float]:
    """
    Smooth box center using causal median (only past + current frames).

    For index 0: return as-is
    For index 1: average of [0, 1]
    For index >= 2: median of [index-2, index-1, index]
    """
    if current_index == 0:
        return centers[0]
    elif current_index == 1:
        return (
            (centers[0][0] + centers[1][0]) / 2.0,
            (centers[0][1] + centers[1][1]) / 2.0,
        )
    else:
        window = centers[max(0, current_index - 2):current_index + 1]
        xs = sorted([c[0] for c in window])
        ys = sorted([c[1] for c in window])
        median_x = xs[len(xs) // 2]
        median_y = ys[len(ys) // 2]
        return (median_x, median_y)


# ══════════════════════════════════════════════════════════════════════
# Coordinate mapping: patch grid → pixel coordinates → crop
# ══════════════════════════════════════════════════════════════════════

def patch_box_to_pixel(
    box_top_left: Tuple[int, int],   # (row, col) in 14×14 patch grid
    box_size: int = 7,               # patch grid box size
    patch_size: int = 16,            # pixels per patch
    image_size: int = 224,           # ViT input size
    pad_size: int = 232,             # padded size
    phase_offset: Tuple[int, int] = (0, 0),  # (dx, dy) which phase this box came from
) -> Tuple[int, int, int, int]:
    """
    Map a patch-grid box to pixel coordinates in the original (unpadded) image.

    The box was computed on a 224×224 crop from the 232×232 padded image.
    We need to account for: phase offset + padding → original coordinates.

    Returns:
        (x_min, y_min, x_max, y_max) in original 224×224 image coordinates.
    """
    row, col = box_top_left
    dx_px, dy_px = phase_offset

    # Patch center → pixel coordinate in the 224 crop
    # Patch (row, col) covers pixels [row*16, (row+1)*16) × [col*16, (col+1)*16)
    x_min_crop = col * patch_size
    y_min_crop = row * patch_size
    x_max_crop = (col + box_size) * patch_size
    y_max_crop = (row + box_size) * patch_size

    # The crop was taken from padded image at offset (dx_px, dy_px)
    # So pixel (x_crop, y_crop) in the crop corresponds to
    # pixel (x_crop + dx_px - 4, y_crop + dy_px - 4) in the original image
    # (because we padded 4px on each side, then shifted by dx_px)
    x_min = x_min_crop + dx_px - 4
    y_min = y_min_crop + dy_px - 4
    x_max = x_max_crop + dx_px - 4
    y_max = y_max_crop + dy_px - 4

    # Clamp to [0, 224)
    x_min = max(0, min(x_min, image_size - 1))
    y_min = max(0, min(y_min, image_size - 1))
    x_max = max(x_min + 1, min(x_max, image_size))
    y_max = max(y_min + 1, min(y_max, image_size))

    return x_min, y_min, x_max, y_max


def expand_and_square_box(
    x_min: int, y_min: int, x_max: int, y_max: int,
    expand_ratio: float = 1.10,
    image_w: int = None, image_h: int = None,
) -> Tuple[int, int, int, int]:
    """
    Expand box by ratio and make it square (for ViT/InternVL input).
    """
    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # Expand
    new_w = w * expand_ratio
    new_h = h * expand_ratio

    # Make square
    side = max(new_w, new_h)
    half = side / 2.0

    x_min_new = int(cx - half)
    y_min_new = int(cy - half)
    x_max_new = int(cx + half)
    y_max_new = int(cy + half)

    # Clamp
    if image_w is not None:
        x_min_new = max(0, x_min_new)
        x_max_new = min(image_w, x_max_new)
    if image_h is not None:
        y_min_new = max(0, y_min_new)
        y_max_new = min(image_h, y_max_new)

    return x_min_new, y_min_new, x_max_new, y_max_new


def map_to_original_resolution(
    x_min: int, y_min: int, x_max: int, y_max: int,
    orig_w: int, orig_h: int,
    vit_input_size: int = 224,
) -> Tuple[int, int, int, int]:
    """
    Map box coordinates from ViT input space (224×224) to original image resolution.

    Assumes the image was resized to 224×224 while preserving aspect ratio
    (with padding on the shorter side).
    """
    # Determine how the image was resized
    scale = min(vit_input_size / orig_w, vit_input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Padding offsets
    pad_x = (vit_input_size - new_w) // 2
    pad_y = (vit_input_size - new_h) // 2

    # Map from 224×224 space to original
    x_min_orig = int((x_min - pad_x) / scale)
    y_min_orig = int((y_min - pad_y) / scale)
    x_max_orig = int((x_max - pad_x) / scale)
    y_max_orig = int((y_max - pad_y) / scale)

    # Clamp
    x_min_orig = max(0, min(x_min_orig, orig_w - 1))
    y_min_orig = max(0, min(y_min_orig, orig_h - 1))
    x_max_orig = max(x_min_orig + 1, min(x_max_orig, orig_w))
    y_max_orig = max(y_min_orig + 1, min(y_max_orig, orig_h))

    return x_min_orig, y_min_orig, x_max_orig, y_max_orig


# ══════════════════════════════════════════════════════════════════════
# Phase feature cache
# ══════════════════════════════════════════════════════════════════════

class PhaseCache:
    """
    Cache per-frame four-phase features across overlapping windows.

    Because window stride (4 frames) << window length (30 frames),
    the same frame appears in multiple windows with different surrounding
    context, requiring different causal smoothing results. We cache the
    raw phase features and raw box (before smoothing) to avoid re-encoding.
    """

    def __init__(self, max_size: int = 256):
        self.cache: dict = {}  # frame_id → {'features': [4,14,14,768], 'valid': [4,14,14], 'raw_box': (r,c), 'raw_energy': float}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, frame_id: int):
        """Return cached data or None."""
        result = self.cache.get(frame_id)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def put(self, frame_id: int, features, valid_mask, raw_box, raw_energy):
        """Store phase data for a frame. Evict oldest if at capacity."""
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache.keys()))
            del self.cache[oldest]
        self.cache[frame_id] = {
            'features': features,
            'valid': valid_mask,
            'raw_box': raw_box,
            'raw_energy': raw_energy,
        }

    def stats(self):
        return {'size': len(self.cache), 'hits': self.hits, 'misses': self.misses}


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== Box search + smooth + crop test ===")

    # Test box search
    H = torch.zeros(14, 14)
    H[3:10, 3:10] = 1.0  # Large high-energy region (7×7)
    valid = torch.ones(14, 14, dtype=torch.bool)
    valid[0, :] = False; valid[-1, :] = False
    valid[:, 0] = False; valid[:, -1] = False

    pos, energy = search_max_energy_box(H, box_size=7, valid_mask=valid)
    print(f"  Box search: pos={pos}, energy={energy:.4f}")
    # The 7×7 box that fully covers the 7×7 region [3:10, 3:10] starts at (3,3)
    assert pos == (3, 3), f"Expected (3,3), got {pos}"

    # Test causal median
    centers = [(3.0, 3.0), (3.5, 3.5), (8.0, 4.0)]  # Simulate a jump at index 2
    smooth_2 = causal_median_center(centers, 2)
    print(f"  Causal median at index 2: {smooth_2}  (expected: median x=3.5, y=3.5)")

    # Test patch→pixel
    box = patch_box_to_pixel((2, 2), box_size=7, phase_offset=(0, 0))
    print(f"  Patch (2,2) → pixel: {box}")

    # Test cache
    cache = PhaseCache(max_size=4)
    cache.put(0, None, None, (2, 2), 1.0)
    cache.put(1, None, None, (3, 3), 0.8)
    print(f"  Cache stats: {cache.stats()}")
    assert cache.get(0) is not None
    assert cache.get(99) is None
    print(f"  After gets: {cache.stats()}")
    print("=== OK! ===")
