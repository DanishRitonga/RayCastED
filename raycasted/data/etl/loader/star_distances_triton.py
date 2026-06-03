"""Triton-accelerated star distance transform for LSP-DETR.

Replaces the Rust star_distances module for training (kept for eval).
Computes (2, n_rays, H, W) distance maps from binary instance masks
entirely on GPU with zero CPU round-trips.
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
    cos_val: tl.constexpr,
    sin_val: tl.constexpr,
    ray_idx: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
):
    """Ray-cast distance from each pixel to mask boundary for ONE ray angle.

    Grid: (tiles_h * tiles_w,)  — one block per (TILE_H, TILE_W) tile.
    Each thread walks along (cos_val, sin_val) until boundary.
    """
    pid = tl.program_id(0)
    tiles_per_row = (H + TILE_H - 1) // TILE_H
    tile_h = pid // tiles_per_row
    tile_w = pid % tiles_per_row

    h_base = tile_h * TILE_H
    w_base = tile_w * TILE_W

    off_h = tl.arange(0, TILE_H)
    off_w = tl.arange(0, TILE_W)
    h = h_base + off_h[:, None]
    w = w_base + off_w[None, :]

    valid = (h < H) & (w < W)
    flat_idx = h * W + w
    mask_val = tl.load(mask_ptr + flat_idx, mask=valid, other=0.0)
    is_inside = mask_val > 0.0

    best_dist = tl.zeros([TILE_H, TILE_W], dtype=tl.float32)
    for step in range(1, MAX_STEPS + 1):
        cur_h_f = (h_base + off_h[:, None]).to(tl.float32) + step.to(tl.float32) * sin_val
        cur_w_f = (w_base + off_w[None, :]).to(tl.float32) + step.to(tl.float32) * cos_val
        cur_h_int = tl.math.floor(cur_h_f).to(tl.int32)
        cur_w_int = tl.math.floor(cur_w_f).to(tl.int32)
        in_bounds = (cur_h_int >= 0) & (cur_h_int < H) & (cur_w_int >= 0) & (cur_w_int < W)
        cur_idx = cur_h_int * W + cur_w_int
        boundary_val = tl.load(mask_ptr + cur_idx, mask=valid & in_bounds, other=0.0)
        hit = is_inside & (boundary_val == 0.0) & (best_dist == 0.0)
        dist_val = (step - 1).to(tl.float32)
        best_dist = tl.where(hit, dist_val, best_dist)

    best_dist = tl.where(is_inside & (best_dist == 0.0), float(MAX_STEPS), best_dist)
    out_idx = ray_idx * H * W + h * W + w
    tl.store(out_ptr + out_idx, best_dist, mask=valid)


def star_distances_triton(
    masks: torch.Tensor,
    n_rays: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU star distance transform from binary instance masks.

    Args:
        masks: (N, H, W) float32 tensor, binary [0/1] or {0, 1}.
        n_rays: Number of radial rays.

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

    TILE_H = 16
    TILE_W = 16
    MAX_STEPS = 400
    tiles_per_row = (H + TILE_H - 1) // TILE_H
    tiles_per_col = (W + TILE_W - 1) // TILE_W
    grid = (tiles_per_row * tiles_per_col,)

    def _run_kernel(mask: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n_rays, H, W, dtype=torch.float32, device=device)
        for r in range(n_rays):
            angle = angles[r]
            _distance_transform_kernel[grid](
                mask,
                out,
                math.cos(angle.item()),
                math.sin(angle.item()),
                r,
                H,
                W,
                MAX_STEPS,
                TILE_H,
                TILE_W,
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
