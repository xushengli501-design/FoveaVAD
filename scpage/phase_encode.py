"""
SC-Page: Sampling-phase instability → local evidence crops.

Phase 1: Four-phase CLIP/ViT encoding
  - Pad 224×224 → 232×232
  - Crop 224×224 at offsets (0,0), (8,0), (0,8), (8,8)
  - Extract ViT patch features [14, 14, 768] per phase
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PhaseEncoder:
    """Four-phase ViT encoding for sampling instability detection."""

    def __init__(
        self,
        model_path: str = '/sdb/data_public/llms/vit-base-patch16-224-in21k',
        device: Optional[torch.device] = None,
    ):
        from transformers import ViTModel

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32  # ViT-B/16 uses float32

        print(f'[PhaseEncoder] Loading ViT-B/16 from {model_path}...', flush=True)
        self.model = ViTModel.from_pretrained(model_path).eval().to(self.device)

        self.patch_size = 16
        self.grid_size = 14  # 224 / 16
        self.feat_dim = 768

        # Phase offsets (half-patch shifts)
        self.offsets = torch.tensor([
            [0, 0], [8, 0], [0, 8], [8, 8],
        ], dtype=torch.long)

        print(f'[PhaseEncoder] Ready: grid={self.grid_size}×{self.grid_size}, dim={self.feat_dim}', flush=True)

    @torch.no_grad()
    def encode_phases(
        self,
        image: torch.Tensor,  # [3, H, W] or PIL Image
    ) -> Dict:
        """
        Extract four-phase ViT patch features from a single image.

        Args:
            image: input image (will be resized to 224×224, then padded to 232×232)

        Returns:
            dict with:
              - features: list of 4 tensors [14, 14, 768], one per phase
              - valid_mask: [4, 14, 14] boolean mask (1=valid, 0=edge/padding)
        """
        # Ensure 224×224 tensor
        if not torch.is_tensor(image):
            from torchvision.transforms import ToTensor
            image = ToTensor()(image)
        if image.shape[-2:] != (224, 224):
            image = F.interpolate(
                image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False,
            ).squeeze(0)

        # Pad to 232×232
        padded = F.pad(image.unsqueeze(0), (4, 4, 4, 4), mode='reflect')  # [1, 3, 232, 232]

        # Extract four phase crops
        phase_images = []
        for dx, dy in self.offsets:
            crop = padded[:, :, dy:dy + 224, dx:dx + 224]  # [1, 3, 224, 224]
            phase_images.append(crop)

        batch = torch.cat(phase_images, dim=0)  # [4, 3, 224, 224]

        # ViT forward
        outputs = self.model(batch.to(device=self.device))
        tokens = outputs.last_hidden_state[:, 1:, :]  # [4, 196, 768] — exclude CLS

        # Reshape to spatial grid
        features = tokens.reshape(4, self.grid_size, self.grid_size, self.feat_dim).float()

        # Build valid mask: exclude the outermost ring (edge artifacts from padding)
        valid_mask = torch.ones(4, self.grid_size, self.grid_size, dtype=torch.bool)
        valid_mask[:, 0, :] = False     # top row
        valid_mask[:, -1, :] = False    # bottom row
        valid_mask[:, :, 0] = False     # left col
        valid_mask[:, :, -1] = False    # right col

        # L2 normalize per-patch features
        features_norm = F.normalize(features, p=2, dim=-1)

        return {
            'features': features_norm,
            'valid_mask': valid_mask,
            'raw_features': features,  # un-normalized, for debugging
        }


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== PhaseEncoder smoke test ===")
    encoder = PhaseEncoder()

    # Dummy image
    dummy = torch.randn(3, 224, 224)
    result = encoder.encode_phases(dummy)

    print(f"  features shape: {result['features'].shape}  (expected: [4, 14, 14, 768])")
    print(f"  valid_mask shape: {result['valid_mask'].shape}  (expected: [4, 14, 14])")
    print(f"  valid cells: {result['valid_mask'][0].sum().item()} / {14*14}")
    assert result['features'].shape == (4, 14, 14, 768)
    assert result['valid_mask'].shape == (4, 14, 14)
    print("=== OK! ===")
