"""Triton-accelerated star distance transform for LSP-DETR.

Replaces the Rust star_distances module for training (kept for eval).
Computes (2, n_rays, H, W) distance maps from binary instance masks
entirely on GPU with zero CPU round-trips.

Algorithm: pixel-walking ray cast from every mask pixel in every direction.
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _distance_transform_kernel(
    mask_ptr,
    out_ptr,
    cos_ptr,
    sin_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    n_rays: tl.constexpr,
    STEP_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    RAYS_PER_BLOCK: tl.constexpr,
):
    MAX_STEPS: tl.constexpr = 400
    pid = tl.program_id(0)
    tiles_per_row = (H + TILE_H - 1) // TILE_H
    num_ray_groups = (n_rays + RAYS_PER_BLOCK - 1) // RAYS_PER_BLOCK

    ray_grp = pid % num_ray_groups
    tile_idx = pid // num_ray_groups
    tile_h = tile_idx // tiles_per_row
    tile_w = tile_idx % tiles_per_row

    h_base = tile_h * TILE_H
    w_base = tile_w * TILE_W

    off_h = tl.arange(0, TILE_H)[:, None, None]
    off_w = tl.arange(0, TILE_W)[None, :, None]
    off_r = tl.arange(0, RAYS_PER_BLOCK)[None, None, :]

    h = h_base + off_h
    w = w_base + off_w
    r = ray_grp * RAYS_PER_BLOCK + off_r

    h_mask = (h < H)[:, :, :]
    w_mask = (w < W)[:, :, :]
    r_mask = (r < n_rays)[:, :, :]
    valid = h_mask & w_mask

    flat_idx = h * W + w
    mask_val = tl.load(mask_ptr + flat_idx, mask=valid, other=0.0)
    is_inside = mask_val > 0.0

    cos = tl.load(cos_ptr + r, mask=r_mask, other=0.0)
    sin = tl.load(sin_ptr + r, mask=r_mask, other=0.0)

    dist = tl.zeros([TILE_H, TILE_W, RAYS_PER_BLOCK], dtype=tl.float32)

    for step in range(1, MAX_STEPS + 1):
        cur_h = h + step * STEP_SIZE * sin
        cur_w = w + step * STEP_SIZE * cos
        cur_h_int = tl.math.llrint(cur_h)
        cur_w_int = tl.math.llrint(cur_w)
        in_bounds = (cur_h_int >= 0) & (cur_h_int < H) & (cur_w_int >= 0) & (cur_w_int < W)
        cur_idx = cur_h_int * W + cur_w_int
        boundary_val = tl.load(mask_ptr + cur_idx, mask=valid & in_bounds, other=0.0)
        hit = is_inside & (boundary_val == 0.0) & (dist == 0.0)
        dist_val = (step - 1) * STEP_SIZE
        dist = tl.where(hit, dist_val, dist)

    dist = tl.where(is_inside & (dist == 0.0), float(MAX_STEPS) * STEP_SIZE, dist)

    out_idx = r * H * W + h * W + w
    tl.store(out_ptr + out_idx, dist, mask=valid & r_mask)


def star_distances_triton(
    masks: torch.Tensor,
    n_rays: int,
    step_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU star distance transform from binary instance masks.

    Args:
        masks: (N, H, W) float32 tensor, binary [0/1].
        n_rays: Number of radial rays.
        step_size: Pixel step size for ray walking (1.0 = 1px accuracy).

    Returns:
        lower: (n_rays, H, W) float32 — tight instance-level lower bound.
        upper: (n_rays, H, W) float32 — distance to any background pixel.
    """
    if masks.numel() == 0 or masks.sum() == 0:
        H = masks.shape[1] if masks.dim() == 3 else 256
        W = masks.shape[2] if masks.dim() == 3 else 256
        z = torch.zeros(n_rays, H, W, dtype=torch.float32, device=masks.device)
        return z, z

    N, H, W = masks.shape
    device = masks.device

    angles = torch.linspace(0, 2 * math.pi, n_rays + 1, device=device)[:n_rays]
    cos_rays = torch.cos(angles)
    sin_rays = torch.sin(angles)

    TILE_H = 16
    TILE_W = 16
    RAYS_PER_BLOCK = 4
    MAX_STEPS = 400  # enough for 256×256 diagonal with 1px step
    num_tiles = ((H + TILE_H - 1) // TILE_H) * ((W + TILE_W - 1) // TILE_W)
    num_ray_groups = (n_rays + RAYS_PER_BLOCK - 1) // RAYS_PER_BLOCK
    grid = (num_tiles * num_ray_groups,)

    def _run_kernel(mask: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n_rays, H, W, dtype=torch.float32, device=device)
        _distance_transform_kernel[grid](
            mask,
            out,
            cos_rays,
            sin_rays,
            H,
            W,
            n_rays,
            step_size,
            TILE_H,
            TILE_W,
            RAYS_PER_BLOCK,
        )
        return out

    any_mask = masks.any(dim=0).float()
    upper = _run_kernel(any_mask)

    lower = torch.full((n_rays, H, W), float('inf'), dtype=torch.float32, device=device)
    for i in range(N):
        inst_mask = masks[i]
        if inst_mask.sum() == 0:
            continue
        inst_dist = _run_kernel(inst_mask)
        lower = torch.min(lower, inst_dist)

    lower = torch.where(torch.isinf(lower), torch.zeros_like(lower), lower)

    return lower, upper
