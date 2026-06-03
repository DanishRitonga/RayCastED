"""Test script: simulate backbone unfreezing with a single synthetic batch."""
from __future__ import annotations

import torch
import torch.nn as nn

from raycasted.model.lsp_detr_model import LSPDetrDetectionModel


def test_backbone_unfreeze():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    nc = 5
    n_rays = 64
    crop_size = 256
    batch_size = 2
    freeze_epochs = 30
    backbone_lr_ratio = 0.1

    print('Creating model...', flush=True)
    model = LSPDetrDetectionModel(nc=nc, n_rays=n_rays, crop_size=crop_size).to(device)
    print('Model created.', flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    bb_params = sum(p.numel() for p in model.backbone.parameters())
    print(f'Total params: {total_params / 1e6:.2f}M, Backbone: {bb_params / 1e6:.2f}M', flush=True)

    # Freeze backbone (as done at init)
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    print(f'\n=== BEFORE UNFREEZE ===', flush=True)
    print(f'Frozen params: {frozen_count}, Trainable params: {trainable_count}', flush=True)
    print(f'Backbone trainable: {any(p.requires_grad for p in model.backbone.parameters())}', flush=True)

    # Optimizer with decoder-only params (matches trainer init)
    wd = 1e-4
    lr = 1e-4
    decay_p, no_decay_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or 'norm' in n.lower():
            no_decay_p.append(p)
        else:
            decay_p.append(p)

    pgs = []
    if decay_p:
        pgs.append({'params': decay_p, 'weight_decay': wd, 'lr': lr, 'name': 'decoder'})
    if no_decay_p:
        pgs.append({'params': no_decay_p, 'weight_decay': 0.0, 'lr': lr, 'name': 'decoder'})

    optimizer = torch.optim.AdamW(pgs)
    print(f'\nOptimizer param groups: {len(optimizer.param_groups)}', flush=True)

    # --- Synthetic batch ---
    print('Creating synthetic batch...', flush=True)
    img = torch.rand(batch_size, 3, crop_size, crop_size, device=device) * 255
    H, W = crop_size, crop_size
    targets = []
    for b in range(batch_size):
        n_inst = 2 + b
        centroids = torch.rand(n_inst, 2)
        rays_norm = torch.rand(n_inst, n_rays) * 0.3
        boxes = torch.cat([centroids, rays_norm], dim=-1).to(device)
        labels = torch.randint(0, nc, (n_inst,), device=device)
        targets.append({
            'labels': labels,
            'boxes': boxes,
            'radial_distances': torch.zeros(2, n_rays, H, W, device=device),
            'centroids': centroids.to(device),
        })

    # Forward + backward (simulate epoch > 0 but < freeze_epochs)
    print(f'\n--- Forward pass (backbone frozen) ---', flush=True)
    model.train()
    use_amp = device.type == 'cuda'
    with torch.amp.autocast('cuda', enabled=use_amp):
        loss, loss_items = model.loss({'img': img, 'targets': targets})
    print(f'Loss: {loss.item():.4f}, CE={loss_items[0]:.4f}, Cent={loss_items[1]:.4f}, Rad={loss_items[2]:.4f}', flush=True)

    loss.backward()
    bb_grad_before = sum(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.backbone.parameters()
    )
    print(f'Backbone params with non-zero grad: {bb_grad_before} (expected 0)', flush=True)
    optimizer.zero_grad()

    # --- Unfreeze backbone (simulate epoch >= freeze_epochs) ---
    print(f'\n=== AFTER UNFREEZE (simulated epoch {freeze_epochs}) ===', flush=True)
    for p in model.backbone.parameters():
        p.requires_grad_(True)

    bb_decay, bb_nodecay = [], []
    for n, p in model.backbone.named_parameters():
        if p.ndim <= 1 or 'norm' in n.lower():
            bb_nodecay.append(p)
        else:
            bb_decay.append(p)

    if bb_decay:
        optimizer.add_param_group({
            'params': bb_decay, 'weight_decay': wd, 'lr': lr * backbone_lr_ratio, 'name': 'backbone',
        })
    if bb_nodecay:
        optimizer.add_param_group({
            'params': bb_nodecay, 'weight_decay': 0.0, 'lr': lr * backbone_lr_ratio, 'name': 'backbone',
        })

    print(f'Optimizer param groups after unfreeze: {len(optimizer.param_groups)}', flush=True)
    for pg in optimizer.param_groups:
        print(f'  {pg["name"]}: lr={pg["lr"]:.2e}, params={len(pg["params"])}', flush=True)

    # Forward + backward after unfreeze
    print(f'\n--- Forward pass (backbone unfrozen) ---', flush=True)
    img2 = torch.rand(batch_size, 3, crop_size, crop_size, device=device) * 255
    with torch.amp.autocast('cuda', enabled=use_amp):
        loss2, loss_items2 = model.loss({'img': img2, 'targets': targets})
    print(f'Loss: {loss2.item():.4f}, CE={loss_items2[0]:.4f}, Cent={loss_items2[1]:.4f}, Rad={loss_items2[2]:.4f}', flush=True)

    loss2.backward()
    bb_grad_after = sum(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.backbone.parameters()
    )
    total_bb = sum(1 for _ in model.backbone.parameters())
    print(f'Backbone params with non-zero grad: {bb_grad_after}/{total_bb} (expected all non-zero)', flush=True)

    # Check no grad explosion
    max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    print(f'Max gradient: {max_grad:.4f}', flush=True)

    # Verify param groups have correct LRs
    decoder_lr = optimizer.param_groups[0]['lr']
    backbone_lr = optimizer.param_groups[-1]['lr']
    assert bb_grad_before == 0, f'Expected 0 backbone grads before unfreeze, got {bb_grad_before}'
    assert bb_grad_after > 0, f'Expected non-zero backbone grads after unfreeze, got {bb_grad_after}'
    assert decoder_lr == lr, f'Expected decoder lr={lr}, got {decoder_lr}'
    assert abs(backbone_lr - lr * backbone_lr_ratio) < 1e-10, \
        f'Expected backbone lr={lr*backbone_lr_ratio}, got {backbone_lr}'

    print(f'\nPASSED: Backbone unfreeze works correctly.', flush=True)
    print(f'  Decoder LR: {decoder_lr:.2e}, Backbone LR: {backbone_lr:.2e}', flush=True)


if __name__ == '__main__':
    test_backbone_unfreeze()
