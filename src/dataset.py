"""PyTorch Dataset for the synthetic TEM nanoparticle images.

Supports the ablation variants (Experiment 3):
  - normalize:    per-image standardization vs. raw [0,255] -> [0,1] scaling
  - use_scale_channel: append the pixel-scale value as a constant 4th channel,
                        so the network sees physical scale explicitly instead
                        of only raw pixels (the "scale-awareness" fix
                        described in the project's scientific note)
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "synthetic_tem"


class NanoparticleDataset(Dataset):
    def __init__(self, split, normalize=True, use_scale_channel=False, metadata=None):
        self.split = split
        self.normalize = normalize
        self.use_scale_channel = use_scale_channel
        meta = metadata if metadata is not None else json.load(open(DATA_DIR / "metadata.json"))
        self.records = [r for r in meta if r["split"] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = np.array(Image.open(DATA_DIR / rec["filename"]), dtype=np.float32)
        img = img / 255.0

        if self.normalize:
            img = (img - img.mean()) / (img.std() + 1e-6)

        # replicate grayscale to 3 channels so we can use ImageNet-pretrained
        # ResNet18 weights as-is
        img3 = np.stack([img, img, img], axis=0)  # (3, H, W)

        scale = rec["pixel_scale_nm_per_px"]
        if self.use_scale_channel:
            scale_channel = np.full_like(img[None], scale)  # (1, H, W)
            img_out = np.concatenate([img3, scale_channel], axis=0)  # (4, H, W)
        else:
            img_out = img3

        return {
            "image": torch.from_numpy(img_out).float(),
            "diameter_nm": torch.tensor(rec["diameter_nm"], dtype=torch.float32),
            "pixel_scale": torch.tensor(scale, dtype=torch.float32),
            "particle_id": rec["particle_id"],
        }
