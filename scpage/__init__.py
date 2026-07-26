"""
SC-Page: Sampling-phase instability → local evidence crops.

For each sampled frame in a 30-frame window:
  1. Four-phase ViT encoding (232→4×224 at half-patch offsets)
  2. Phase instability map H(p) with local MAD normalization
  3. 7×7 box search on 14×14 heatmap
  4. Causal median smoothing of box centers
  5. Patch→pixel coordinate mapping → crop from original frame
  6. Cross-window phase feature caching

Returns per-window: global_images [8], local_crops [8]
"""

try:
    from scpage.phase_encode import PhaseEncoder
    from scpage.instability import compute_phase_instability
    from scpage.crop_utils import (
        search_max_energy_box,
        causal_median_center,
        patch_box_to_pixel,
        expand_and_square_box,
        map_to_original_resolution,
        PhaseCache,
    )
except ImportError:
    from phase_encode import PhaseEncoder
    from instability import compute_phase_instability
    from crop_utils import (
        search_max_energy_box,
        causal_median_center,
        patch_box_to_pixel,
        expand_and_square_box,
        map_to_original_resolution,
        PhaseCache,
    )

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class SCPageExtractor:
    """Full SC-Page extraction pipeline for one window."""

    def __init__(
        self,
        vit_path: str = '/sdb/data_public/llms/vit-base-patch16-224-in21k',
        device: Optional[torch.device] = None,
        box_size: int = 7,
        expand_ratio: float = 1.10,
        cache_size: int = 256,
    ):
        self.encoder = PhaseEncoder(vit_path, device)
        self.device = self.encoder.device
        self.box_size = box_size
        self.expand_ratio = expand_ratio
        self.cache = PhaseCache(max_size=cache_size)

        # Phase offsets in pixels (for coordinate mapping)
        self.phase_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]

    @torch.no_grad()
    def process_frame(
        self,
        image: torch.Tensor,        # [3, H, W] in [0, 1]
        frame_id: int,
        prev_centers: List[Tuple[float, float]],  # box centers from previous frames in this window
        current_index: int,          # position in the 8-frame sample sequence
        orig_size: Tuple[int, int] = None,  # (W, H) of original frame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one frame: compute instability → find crop region.

        Args:
            image: input frame tensor [3, H, W]
            frame_id: global frame index (for caching)
            prev_centers: list of (cx, cy) from previous frames in window
            current_index: 0..7 position in the 8-frame sample
            orig_size: original frame (W, H) for coordinate mapping

        Returns:
            global_img: the full frame resized to 448×448 (for InternVL)
            local_crop: the cropped region resized to 448×448
        """
        device = self.device

        # Check cache for phase features
        cached = self.cache.get(frame_id)

        if cached is None:
            # Full phase encoding
            result = self.encoder.encode_phases(image)
            features = result['features']      # [4, 14, 14, 768]
            valid_mask = result['valid_mask']   # [4, 14, 14]

            # Compute instability map
            H, D_raw = compute_phase_instability(features, valid_mask)

            # Search for max-energy 7×7 box
            valid_search = valid_mask[0].clone().to(device)
            (box_row, box_col), energy = search_max_energy_box(
                H, box_size=self.box_size, valid_mask=valid_search,
            )

            # Store in cache
            self.cache.put(frame_id, features.cpu(), valid_mask.cpu(),
                          (box_row, box_col), energy)
        else:
            features = cached['features'].to(device)
            valid_mask = cached['valid'].to(device)
            box_row, box_col = cached['raw_box']
            energy = cached['raw_energy']

        # Causal median smoothing
        raw_center = (box_col + self.box_size / 2.0, box_row + self.box_size / 2.0)
        all_centers = prev_centers + [raw_center]
        smooth_center = causal_median_center(all_centers, current_index)

        # Map to pixel coordinates in ViT input space (224×224)
        x_min, y_min, x_max, y_max = patch_box_to_pixel(
            (box_row, box_col),
            box_size=self.box_size,
            phase_offset=(0, 0),  # Use phase-0 coordinates
        )

        # Center on smoothed position
        cx_smooth, cy_smooth = smooth_center
        w_box = x_max - x_min
        h_box = y_max - y_min
        x_min_s = int(cx_smooth - w_box / 2.0)
        x_max_s = int(cx_smooth + w_box / 2.0)
        y_min_s = int(cy_smooth - h_box / 2.0)
        y_max_s = int(cy_smooth + h_box / 2.0)

        # Expand + square
        x_min_e, y_min_e, x_max_e, y_max_e = expand_and_square_box(
            x_min_s, y_min_s, x_max_s, y_max_s,
            expand_ratio=self.expand_ratio,
            image_w=224, image_h=224,
        )

        # Map to original resolution
        if orig_size is not None:
            orig_w, orig_h = orig_size
            x_min_o, y_min_o, x_max_o, y_max_o = map_to_original_resolution(
                x_min_e, y_min_e, x_max_e, y_max_e,
                orig_w=orig_w, orig_h=orig_h,
            )
        else:
            # Assume the image is already 224×224
            x_min_o, y_min_o, x_max_o, y_max_o = x_min_e, y_min_e, x_max_e, y_max_e

        # Produce global image (448×448 for InternVL) and local crop (448×448)
        global_img = _resize_for_internvl(image, 448)

        if orig_size is not None:
            local_crop = _crop_and_resize(image, x_min_o, y_min_o, x_max_o, y_max_o, 448)
        else:
            local_crop = _crop_and_resize(image, x_min_e, y_min_e, x_max_e, y_max_e, 448)

        return global_img, local_crop, smooth_center

    @torch.no_grad()
    def process_window(
        self,
        sampled_frames: List[torch.Tensor],   # [8] list of [3, H, W]
        frame_ids: List[int],                  # [8] global frame indices
        orig_sizes: List[Tuple[int, int]] = None,  # [8] original (W, H)
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Process all 8 sampled frames in a window.

        Returns:
            global_images: [8] 448×448 tensors
            local_crops: [8] 448×448 tensors
        """
        global_images = []
        local_crops = []
        centers = []  # [(cx, cy), ...] history for causal smoothing

        for j, image in enumerate(sampled_frames):
            orig_size = orig_sizes[j] if orig_sizes else None
            global_img, local_crop, center = self.process_frame(
                image, frame_ids[j], centers, j, orig_size,
            )
            global_images.append(global_img)
            local_crops.append(local_crop)
            centers.append(center)

        return global_images, local_crops


def _resize_for_internvl(image: torch.Tensor, size: int = 448) -> torch.Tensor:
    """Resize image for InternVL input."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    return F.interpolate(image, size=(size, size), mode='bilinear', align_corners=False)


def _crop_and_resize(
    image: torch.Tensor,  # [3, H, W]
    x_min: int, y_min: int, x_max: int, y_max: int,
    target_size: int = 448,
) -> torch.Tensor:
    """Crop region and resize to target_size."""
    if image.dim() == 3:
        crop = image[:, y_min:y_max, x_min:x_max]
    else:
        crop = image[:, :, y_min:y_max, x_min:x_max]
    return _resize_for_internvl(crop, target_size)


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== SCPageExtractor smoke test ===")
    extractor = SCPageExtractor()

    # Create 8 dummy frames
    dummy_frames = [torch.randn(3, 224, 224).clamp(0, 1) for _ in range(8)]
    frame_ids = list(range(8))

    global_imgs, local_crops = extractor.process_window(dummy_frames, frame_ids)

    print(f"  global_images: {len(global_imgs)} frames, shape={global_imgs[0].shape}")
    print(f"  local_crops: {len(local_crops)} frames, shape={local_crops[0].shape}")
    print(f"  cache: {extractor.cache.stats()}")
    assert len(global_imgs) == 8
    assert len(local_crops) == 8
    print("=== OK! ===")
