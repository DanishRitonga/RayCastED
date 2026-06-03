"""GPU batch augmentation replacing albumentations pipeline.

Drops Superpixels and ZoomBlur (combined p=0.19, not GPU-friendly).
Normalize stays in model forward (original LSP-DETR design).

All transforms run on GPU in a single batch-level forward pass.
Images: (B, 3, H, W) float32 [0, 255]
Masks:  (B, N, H, W) float32 [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPUAugment(nn.Module):
    def __init__(self, crop_size: int = 256) -> None:
        super().__init__()
        self.crop_size = crop_size

    def _sample_params(self, B: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Generate per-sample random parameters for all transforms."""
        p = {
            'flip_h': torch.rand(B, device=device),
            'flip_v': torch.rand(B, device=device),
            'rotate_k': torch.randint(0, 4, (B,), device=device).float(),
            'downscale': torch.rand(B, device=device),
            'blur': torch.rand(B, device=device),
            'gauss_noise': torch.rand(B, device=device),
            'color_jitter': torch.rand(B, device=device),
            'sized_crop': torch.rand(B, device=device),
            'elastic': torch.rand(B, device=device),
        }
        # Sized crop params
        p['crop_h'] = torch.randint(128, 257, (B,), device=device).float()
        p['crop_top'] = torch.rand(B, device=device) * (256 - p['crop_h'])
        p['crop_left'] = torch.rand(B, device=device) * (256 - p['crop_h'])
        # Blur kernel size (odd, 3-11)
        p['blur_ksize'] = (torch.randint(1, 6, (B,), device=device) * 2 + 1).float()
        # GaussNoise std
        p['noise_std'] = torch.rand(B, device=device) * 0.44
        # ColorJitter factors
        p['cj_brightness'] = 1.0 + (torch.rand(B, device=device) * 2 - 1) * 0.25
        p['cj_contrast'] = 1.0 + (torch.rand(B, device=device) * 2 - 1) * 0.25
        p['cj_saturation'] = 1.0 + (torch.rand(B, device=device) * 2 - 1) * 0.1
        p['cj_hue'] = (torch.rand(B, device=device) * 2 - 1) * 0.05
        return p

    @staticmethod
    def _apply_geom(
        images: torch.Tensor,
        masks: torch.Tensor,
        params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply flip, rotate90, downscale, sized_crop as a single GPU pass."""
        B, _, H, W = images.shape

        # Per-sample transform application (loop over batch for per-sample params)
        out_images = []
        out_masks = []
        for b in range(B):
            img = images[b : b + 1]
            msk = masks[b : b + 1]

            # Flip horizontal (p=0.5)
            if params['flip_h'][b] < 0.5:
                img = torch.flip(img, dims=[-1])
                msk = torch.flip(msk, dims=[-1])

            # Flip vertical (p=0.5)
            if params['flip_v'][b] < 0.5:
                img = torch.flip(img, dims=[-2])
                msk = torch.flip(msk, dims=[-2])

            # Rotate 90° (p=1.0, always applied, uniformly {0,1,2,3})
            k = int(params['rotate_k'][b].item())
            if k > 0:
                img = torch.rot90(img, k, dims=[-2, -1])
                msk = torch.rot90(msk, k, dims=[-2, -1])

            # Downscale (p=0.15, scale=0.5)
            if params['downscale'][b] < 0.15:
                ds = max(H // 2, 32)
                img = F.interpolate(img, size=(ds, ds), mode='bilinear', align_corners=False)
                msk = F.interpolate(msk, size=(ds, ds), mode='nearest')
                img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
                msk = F.interpolate(msk, size=(H, W), mode='nearest')

            # RandomSizedCrop (p=0.1, 128-256 px → resize to 256)
            if params['sized_crop'][b] < 0.1:
                ch = int(params['crop_h'][b].item())
                cw = ch
                top = int(params['crop_top'][b].item())
                left = int(params['crop_left'][b].item())
                img = img[:, :, top : top + ch, left : left + cw]
                msk = msk[:, :, top : top + ch, left : left + cw]
                img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
                msk = F.interpolate(msk, size=(H, W), mode='nearest')

            out_images.append(img)
            out_masks.append(msk)

        return torch.cat(out_images, dim=0), torch.cat(out_masks, dim=0)

    @staticmethod
    def _gauss_noise(images: torch.Tensor, params: dict) -> torch.Tensor:
        """Add Gaussian noise (p=0.25, std 0-0.44)."""
        B = images.shape[0]
        mask = params['gauss_noise'] < 0.25
        if mask.any():
            noise = torch.randn_like(images[mask]) * params['noise_std'][mask].view(-1, 1, 1, 1)
            images = images.clone()
            images[mask] = images[mask] + noise
        return images

    @staticmethod
    def _box_blur(images: torch.Tensor, params: dict) -> torch.Tensor:
        """Box blur (p=0.2, kernel 3-11 odd via depthwise conv2d)."""
        B, C, H, W = images.shape
        mask = params['blur'] < 0.2
        if not mask.any():
            return images
        images = images.clone()
        for b in range(B):
            if mask[b]:
                k = int(params['blur_ksize'][b].item())
                kernel = torch.ones(1, 1, k, k, device=images.device) / (k * k)
                img_b = images[b : b + 1]
                for c in range(C):
                    images[b, c : c + 1] = F.conv2d(
                        F.pad(img_b[:, c : c + 1], (k // 2,) * 4, mode='reflect'),
                        kernel,
                    )
        return images

    @staticmethod
    def _color_jitter(images: torch.Tensor, params: dict) -> torch.Tensor:
        """Color jitter: brightness + contrast + saturation + hue (p=0.2)."""
        B = images.shape[0]
        mask = params['color_jitter'] < 0.2
        if not mask.any():
            return images
        images = images.clone()
        for b in range(B):
            if not mask[b]:
                continue
            img = images[b]  # (3, H, W)

            # Brightness
            img = img * params['cj_brightness'][b]

            # Contrast: blend with grayscale mean
            if params['cj_contrast'][b] != 1.0:
                gray = img.mean(dim=0, keepdim=True)
                img = gray + (img - gray) * params['cj_contrast'][b]

            # Saturation: blend with grayscale
            if params['cj_saturation'][b] != 1.0:
                gray = img.mean(dim=0, keepdim=True)
                img = gray + (img - gray) * params['cj_saturation'][b]

            # Hue: rotate in RGB space (approximation via XYZ-like rotation)
            hue_factor = params['cj_hue'][b]
            if hue_factor != 0.0:
                h_angle = hue_factor * 3.14159265
                u = torch.cos(torch.as_tensor(h_angle, device=img.device))
                w = torch.sin(torch.as_tensor(h_angle, device=img.device))
                # Approximate hue rotation matrix
                rot = torch.tensor(
                    [
                        [0.299 + 0.701 * u + 0.168 * w, 0.587 - 0.587 * u + 0.330 * w, 0.114 - 0.114 * u - 0.497 * w],
                        [0.299 - 0.299 * u - 0.328 * w, 0.587 + 0.413 * u + 0.035 * w, 0.114 - 0.114 * u + 0.292 * w],
                        [0.299 - 0.300 * u + 1.250 * w, 0.587 - 0.588 * u - 1.050 * w, 0.114 + 0.886 * u - 0.203 * w],
                    ],
                    device=img.device,
                )
                img = torch.einsum('chw,cd->dhw', img, rot)

            images[b] = img.clamp(0, 255)
        return images

    @staticmethod
    def _elastic(images: torch.Tensor, masks: torch.Tensor, params: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Elastic transform (p=0.2, sigma=25, alpha=0.5)."""
        mask = params['elastic'] < 0.2
        if not mask.any():
            return images, masks
        B, _, H, W = images.shape
        for b in range(B):
            if not mask[b]:
                continue
            # Generate displacement fields
            sigma = 25
            alpha = 0.5
            kernel_size = int(4 * sigma + 1) | 1
            dx = torch.randn(1, 2, H, W, device=images.device) * alpha
            dx = F.avg_pool2d(F.pad(dx, (kernel_size // 2,) * 4, mode='reflect'), kernel_size, 1, 0)
            dy = torch.randn(1, 2, H, W, device=images.device) * alpha
            dy = F.avg_pool2d(F.pad(dy, (kernel_size // 2,) * 4, mode='reflect'), kernel_size, 1, 0)
            dy_coords = dy.permute(0, 2, 3, 1)

            # Grid sample
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=images.device),
                torch.linspace(-1, 1, W, device=images.device),
                indexing='ij',
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
            grid = grid + dy_coords * 0.02

            images[b : b + 1] = F.grid_sample(images[b : b + 1], grid, mode='bilinear', align_corners=False)
            masks[b : b + 1] = F.grid_sample(masks[b : b + 1], grid, mode='nearest', align_corners=False)

        return images, masks

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = images.shape
        params = self._sample_params(B, images.device)

        images, masks = self._apply_geom(images, masks, params)
        images = self._gauss_noise(images, params)
        images = self._box_blur(images, params)
        images = self._color_jitter(images, params)
        images, masks = self._elastic(images, masks, params)

        return images, masks
