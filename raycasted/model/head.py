"""RayCastED — RayCast Detection Head.

Implements the RayRefinementBlock and RayCastDetect head that replaces
YOLO's bounding-box regression with 34-dim raycast polygon predictions.

Output format: [xy_offset(2), rays(32)] per anchor.
  - xy_offset: Sigmoid-activated, grid-cell-relative, bounded [0, 1]
  - rays: Softplus-activated, strictly positive distances

DFL is completely removed — it has no meaning for polar coordinates.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors

RAYCAST_DIM = 34  # xy_offset(2) + rays(32)


class RayRefinementBlock(nn.Module):
    """3x3 depthwise conv + GroupNorm + SiLU + residual skip.

    Blends features from spatially neighbouring anchor points before
    committing to ray predictions — reduces jagged outputs without
    Transformer overhead (CPP-Net design).
    """

    def __init__(self, channels: int):
        super().__init__()
        num_groups = 4 if channels < 64 else 8
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.gn = nn.GroupNorm(num_groups, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise conv refinement with residual connection."""
        return x + self.act(self.gn(self.dwconv(x)))


class RayCastDetect(Detect):
    """Polygon detection head replacing bounding-box regression with raycast.

    Subclasses ultralytics Detect, replacing the cv2 regression branch with:
        Conv → Conv → RayRefinementBlock → Conv2d(c2, 34, 1)

    DFL is removed entirely. Output activations:
        - Channels 0-1 (xy): Sigmoid (inference only)
        - Channels 2-33 (rays): Softplus (inference only)
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        """Initialize polygon detection head.

        Args:
            nc: Number of classes.
            reg_max: DFL channels (ignored for polygon head, kept for API compat).
            end2end: Whether to use end-to-end NMS-free detection.
            ch: Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)

        # BUG-02 fix: override self.no from nc + reg_max*4 to nc + 34
        self.no = nc + RAYCAST_DIM

        # Replace cv2 (box regression) with polygon regression stack
        c2 = max(16, ch[0] // 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                RayRefinementBlock(c2),
                nn.Conv2d(c2, RAYCAST_DIM, 1),
            )
            for x in ch
        )

        # Remove DFL — not applicable to polygon regression
        self.dfl = nn.Identity()

        # Recreate one2one heads with polygon cv2
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Concatenate polygon predictions and class scores across scales.

        Returns dict with 'boxes' key containing 34-dim polygon logits
        and 'scores' key containing class logits.
        """
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        poly = torch.cat([box_head[i](x[i]).view(bs, RAYCAST_DIM, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=poly, scores=scores, feats=x)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode polygon predictions for inference.

        Applies Sigmoid to xy offsets and Softplus to ray distances.
        Skips DFL entirely.
        """
        shape = x['feats'][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x['feats'], self.stride, 0.5))
            self.shape = shape

        poly = x['boxes']  # [B, 34, N_anchors]
        xy_offset = poly[:, :2, :].sigmoid()  # bounded [0, 1]
        rays = F.softplus(poly[:, 2:, :])  # strictly positive

        # Decode xy from grid-relative to absolute coords
        xy_abs = (xy_offset * 2.0 - 0.5 + self.anchors) * self.strides
        rays_abs = rays * self.strides

        dbox = torch.cat([xy_abs, rays_abs], dim=1)
        return torch.cat((dbox, x['scores'].sigmoid()), 1)

    def bias_init(self):
        """Initialize polygon head biases.

        XY channels (0-1): bias=2.0 → sigmoid(2.0)≈0.88, encourages initial detections.
        Ray channels (2-33): calibrated for ~15px radius cells at 0.25 MPP
            (lymphocytes 7-10μm → 28-40px diameter → radius ≈15px).
            Inverse softplus: bias = log(exp(target_px / stride) - 1).
        """
        target_ray_px = 15.0  # reasonable lymphocyte radius at 0.25 MPP
        for i, (a, b) in enumerate(zip(self.one2many['box_head'], self.one2many['cls_head'])):
            bias = a[-1].bias.data
            bias[:2] = 2.0  # xy: sigmoid → ~0.88
            bias[2:] = math.log(math.exp(target_ray_px / self.stride[i]) - 1)  # rays → ~15px
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one['box_head'], self.one2one['cls_head'])):
                bias = a[-1].bias.data
                bias[:2] = 2.0
                bias[2:] = math.log(math.exp(target_ray_px / self.stride[i]) - 1)
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
