"""Diagnostic script — inspect checkpoint and model output for eval debugging."""

import numpy as np
import torch

from raycasted.model.register import register_raycast_head

WEIGHTS = 'runs/detect/train5/weights/best.pt'

register_raycast_head()
ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
model = ckpt['model'] if isinstance(ckpt, dict) else ckpt
model = model.float().eval()

# Check head config
head = model.model[-1]
print(f'nc={head.nc}, n_rays={head.n_rays}, raycast_dim={head.raycast_dim}')
print(f'training_args={getattr(model, "training_args", {})}')

# Test forward pass on random input
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out = model(x)

decoded = out[0]  # [1, max_det, raycast_dim+2]
print(f'\nOutput shape: {decoded.shape}')

# Confidence distribution
raycast_dim = head.raycast_dim
confs = decoded[0, :, raycast_dim].numpy()
print(f'Conf > 0.2: {(confs > 0.2).sum()}/{len(confs)}')
print(f'Conf > 0.5: {(confs > 0.5).sum()}/{len(confs)}')
print(f'Conf > 0.8: {(confs > 0.8).sum()}/{len(confs)}')
print(f'Conf distribution: min={confs.min():.4f}, median={np.median(confs):.4f}, max={confs.max():.4f}')

# Class distribution
cls_ids = decoded[0, :, raycast_dim + 1].numpy().astype(int)
unique, counts = np.unique(cls_ids, return_counts=True)
print('\nClass distribution (top-10):')
for c, n in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
    print(f'  class {c}: {n} predictions')

# Show first 5 predictions
print('\nFirst 5 predictions (cx, cy, conf, cls):')
for i in range(min(5, decoded.shape[1])):
    row = decoded[0, i].numpy()
    print(f'  [{i}] cx={row[0]:.1f}, cy={row[1]:.1f}, conf={row[raycast_dim]:.4f}, cls={int(row[raycast_dim + 1])}')
