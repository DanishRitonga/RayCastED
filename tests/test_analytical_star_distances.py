"""Smoke test: RT-DETR with analytical star-distance loss on synthetic data.

Verifies the full training pipeline (forward + loss + backward) with
Cramer's rule analytical ray-polygon intersection for the matcher cost
and loss targets.
"""

import torch

from raycasted.data.etl.utils import constants as _const
from raycasted.model.blocks.star_distances_analytical import _normed_rays_to_vertices, build_ray_directions


def _reconstruct_polygon_vertices(bboxes_norm, crop_size, n_rays=None):
    """Reconstruct pixel-space polygon vertices from normalized ray vectors."""
    if n_rays is None:
        n_rays = bboxes_norm.shape[1] - 2
    centroids_px = bboxes_norm[:, :2] * crop_size
    rays_norm = bboxes_norm[:, 2 : 2 + n_rays]
    ray_cos, ray_sin = build_ray_directions(n_rays)
    return _normed_rays_to_vertices(centroids_px, rays_norm, crop_size, ray_cos, ray_sin)


def _make_circular_nuclei(n_cells, n_rays, crop_size=256.0, seed=42):
    """Create synthetic circular nucleus annotations in normalized space."""
    gen = torch.Generator().manual_seed(seed)

    cx = torch.rand(n_cells, generator=gen) * 0.6 + 0.2
    cy = torch.rand(n_cells, generator=gen) * 0.6 + 0.2
    radius = torch.rand(n_cells, generator=gen) * 0.03 + 0.01

    cls = torch.zeros(n_cells, dtype=torch.float32)
    rays = radius.unsqueeze(1).repeat(1, n_rays)

    bboxes = torch.cat([cx.unsqueeze(1), cy.unsqueeze(1), rays], dim=1)
    return cls, bboxes


def _make_batch(bs=1, n_cells=8, crop_size=256, n_rays=64):
    """Build a full batch dict with images, annotations, and pixel-space vertices."""
    cls_list, bboxes_list, vertices_list = [], [], []
    for _ in range(bs):
        cl, bb = _make_circular_nuclei(n_cells, n_rays, crop_size)
        verts = _reconstruct_polygon_vertices(bb, crop_size, n_rays)
        cls_list.append(cl)
        bboxes_list.append(bb)
        vertices_list.append(verts)

    batch_cls = torch.cat(cls_list, dim=0)
    batch_bboxes = torch.cat(bboxes_list, dim=0)
    batch_vertices = torch.cat(vertices_list, dim=0)
    batch_idx = torch.zeros(batch_cls.shape[0], dtype=torch.float32)

    img = torch.randn(bs, 3, crop_size, crop_size, dtype=torch.float32)
    ori_shape = torch.full((bs, 2), crop_size)
    ratio_pad = [(torch.ones(1, 1), torch.zeros(1, 2)) for _ in range(bs)]

    return {
        'img': img,
        'batch_idx': batch_idx,
        'cls': batch_cls,
        'bboxes': batch_bboxes,
        'gt_vertices': batch_vertices,
        'ori_shape': ori_shape,
        'ratio_pad': ratio_pad,
        'im_file': [f'tile_{i:04d}.npz' for i in range(bs)],
    }


def test_analytical_star_distances_smoke():
    """End-to-end smoke test: forward + loss + backward with RT-DETR."""
    _const.configure_rays(64)

    from raycasted.model.rtdetr_model import RayCastRTDETRDetectionModel

    cfg = 'raycasted/cfg/yolo26s-rtdetr-p234.yaml'
    model = RayCastRTDETRDetectionModel(cfg, ch=3, nc=5, verbose=False)
    model.train()

    batch = _make_batch(bs=1, n_cells=5, crop_size=256, n_rays=64)

    loss_val, loss_items = model.loss(batch)
    assert torch.isfinite(loss_val), f'loss is not finite: {loss_val}'
    assert loss_val.item() > 0, f'loss should be positive, got {loss_val.item()}'

    loss_val.backward()

    total_params = 0
    grad_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_params += p.numel()
            if p.grad is not None:
                grad_params += p.numel()

    assert grad_params > 0, 'no parameters received gradients'
    assert grad_params / total_params > 0.5, f'only {grad_params}/{total_params} params have gradients'


def test_analytical_ray_cost_shape():
    """Verify analytical ray cost produces correct [n_pred, n_gt] shape."""
    _const.configure_rays(64)

    from raycasted.model.blocks.star_distances_analytical import (
        analytical_ray_cost,
        build_ray_directions,
    )

    pred_xy = torch.rand(10, 2) * 256.0
    vertices = torch.rand(5, 64, 2) * 256.0
    ray_cos, ray_sin = build_ray_directions(64)

    cost = analytical_ray_cost(pred_xy, vertices, ray_cos, ray_sin)
    assert cost.shape == (10, 5), f'expected (10,5), got {cost.shape}'
    assert torch.isfinite(cost).all(), 'cost contains inf/nan'
    assert (cost >= 0).all(), 'cost contains negatives'


def test_analytical_gt_rays_diagonal():
    """Verify analytical_gt_rays returns [N, n_rays] not [N, N, n_rays]."""
    _const.configure_rays(64)

    from raycasted.model.blocks.star_distances_analytical import (
        analytical_gt_rays,
        build_ray_directions,
    )

    n = 3
    _, bboxes = _make_circular_nuclei(n, n_rays=64, crop_size=256.0)
    pred_xy = bboxes[:, :2] * 256.0 + torch.randn(n, 2) * 10.0
    vertices = _reconstruct_polygon_vertices(bboxes, crop_size=256.0)
    ray_cos, ray_sin = build_ray_directions(64)

    gt_rays = analytical_gt_rays(pred_xy, vertices, ray_cos, ray_sin, crop_size=256.0)
    assert gt_rays.shape == (n, 64), f'expected ({n},64), got {gt_rays.shape}'
    assert torch.isfinite(gt_rays).all(), 'gt_rays contains inf/nan'
    assert (gt_rays >= 0).all(), 'gt_rays contains negatives'
