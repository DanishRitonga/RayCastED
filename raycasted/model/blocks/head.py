"""RayCastED — RayCast Detection Head.

Implements the RayRefinementBlock and RayCastDetect head that replaces
YOLO's bounding-box regression with raycast polygon predictions.

Output format: [xy_offset(2), rays(n_rays)] per anchor.
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

# Default raycast dimension for backward compat (tests, export).
# At runtime, use head.raycast_dim which reflects the actual n_rays parameter.
from raycasted.data.etl.utils import constants as _const

RAYCAST_DIM = 2 + _const.N_RAYS


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


class LargeKernelRefinementBlock(nn.Module):
    """Large-kernel depthwise conv with reparameterizable small-kernel branch.

    Based on LKCell / RepLKNet design:
      - Primary branch: large depthwise conv (e.g., 7x7 or 13x13)
      - Auxiliary branch: small depthwise conv (e.g., 3x3 or 5x5)
      - At deploy time, merge small kernel into large kernel via reparameterization

    The large kernel gives each anchor a wider receptive field to see
    neighbouring cells and boundaries, improving polygon quality for
    dense touching cells without Transformer overhead.
    """

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        num_groups = 4 if channels < 64 else 8
        assert kernel_size % 2 == 1, f'kernel_size must be odd, got {kernel_size}'
        padding = kernel_size // 2

        self.lk_conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.lk_gn = nn.GroupNorm(num_groups, channels)

        # Small auxiliary kernel: use 5 for large kernels, 3 for kernel_size=7
        small_k = 5 if kernel_size >= 11 else 3
        small_pad = small_k // 2
        self.sk_conv = nn.Conv2d(channels, channels, small_k, padding=small_pad, groups=channels)
        self.sk_gn = nn.GroupNorm(num_groups, channels)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual-branch large-kernel refinement with residual."""
        lk = self.act(self.lk_gn(self.lk_conv(x)))
        sk = self.act(self.sk_gn(self.sk_conv(x)))
        return x + lk + sk

    def reparameterize(self):
        """Merge small-kernel branch into large-kernel for deployment.

        Fuses sk_conv weights into lk_conv via zero-padding, then merges
        GroupNorm+conv for each branch. After calling this, sk_conv/sk_gn
        can be removed to reduce inference cost.
        """
        sk_weight = self.sk_conv.weight.data
        sk_bias = self.sk_conv.bias.data
        lk_weight = self.lk_conv.weight.data
        lk_bias = self.lk_conv.bias.data

        sk_k = sk_weight.shape[-1]
        lk_k = lk_weight.shape[-1]
        pad = (lk_k - sk_k) // 2
        sk_padded = F.pad(sk_weight, [pad] * 4)

        self.lk_conv.weight.data.copy_(lk_weight + sk_padded)
        self.lk_conv.bias.data.copy_(lk_bias + sk_bias)


class RayCastDetect(Detect):
    """Polygon detection head replacing bounding-box regression with raycast.

    Subclasses ultralytics Detect, replacing the cv2 regression branch with:
        Conv → Conv → RayRefinementBlock → Conv2d(c2, raycast_dim, 1)

    DFL is removed entirely. Output activations:
        - Channels 0-1 (xy): Sigmoid (inference only)
        - Channels 2..(2+n_rays) (rays): Softplus (inference only)
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        n_rays: int | None = None,
        head_channel_scale: float = 0.5,
        head_channel_min: int = 64,
        refinement_kernel_size: int = 3,
        aux_xy: bool = False,
    ):
        """Initialize polygon detection head.

        Args:
            nc: Number of classes.
            reg_max: DFL channels (ignored for polygon head, kept for API compat).
            end2end: Whether to use end-to-end NMS-free detection.
            ch: Tuple of channel sizes from backbone feature maps.
            n_rays: Number of radial rays for polygon parameterization.
            head_channel_scale: Fraction of input channels for head width.
            head_channel_min: Minimum head intermediate channels.
            refinement_kernel_size: Kernel size for polygon refinement block.
                3 = standard RayRefinementBlock (default).
                7 or 13 = LargeKernelRefinementBlock (LKCell-style, wider receptive field).
            aux_xy: If True, attach a lightweight 1x1 conv head for auxiliary xy regression
                directly on neck features, bypassing the 4-layer head stack.
        """
        self.n_rays = n_rays if n_rays is not None else _const.N_RAYS
        self.raycast_dim = 2 + self.n_rays  # xy + rays
        self._end2end_arg = end2end  # store before parent __init__ (end2end is a property)

        super().__init__(nc, reg_max, end2end, ch)

        # BUG-02 fix: override self.no from nc + reg_max*4 to nc + raycast_dim
        self.no = nc + self.raycast_dim

        # Select refinement block based on kernel size
        if refinement_kernel_size <= 3:
            block_cls = RayRefinementBlock
            block_kwargs = {}
        else:
            block_cls = LargeKernelRefinementBlock
            block_kwargs = {'kernel_size': refinement_kernel_size}

        # Replace cv2 (box regression) with polygon regression stack
        c2 = max(head_channel_min, int(ch[0] * head_channel_scale))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                block_cls(c2, **block_kwargs),
                nn.Conv2d(c2, self.raycast_dim, 1),
            )
            for x in ch
        )

        # Remove DFL — not applicable to polygon regression
        self.dfl = nn.Identity()

        # Auxiliary xy head: lightweight 1x1 conv directly on neck features
        # Bypasses the 4-layer head stack to give backbone direct xy gradient
        if aux_xy:
            self.aux_xy = nn.ModuleList(nn.Conv2d(c, 2, 1) for c in ch)
            for layer in self.aux_xy:
                nn.init.zeros_(layer.bias)
                nn.init.zeros_(layer.weight)
        else:
            self.aux_xy = None

        # Separate o2o heads — both box (cv2) and cls (cv3) are deepcopied.
        # Shared cv3 caused 11.5x overprediction: o2m's dense positives (topk=15)
        # taught the shared cls head to fire high scores for many anchors per GT,
        # defeating o2o's topk2=1 duplicate suppression.
        if self._end2end_arg:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            # one2one_cv3 is created by parent Detect.__init__ as deepcopy of cv3

    @property
    def one2many(self):
        """Return one2many head components."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Return one2one head components — separate cls head for NMS-free inference."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Concatenate polygon predictions and class scores across scales.

        Returns dict with 'boxes' key containing raycast_dim polygon logits,
        'scores' key containing class logits, and 'feats' key with feature maps.
        """
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        poly = torch.cat([box_head[i](x[i]).view(bs, self.raycast_dim, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        result = dict(boxes=poly, scores=scores, feats=x)

        if hasattr(self, 'aux_xy') and self.aux_xy is not None:
            aux_raw = torch.cat([self.aux_xy[i](x[i]).view(bs, 2, -1) for i in range(self.nl)], dim=-1)
            result['aux_xy_raw'] = aux_raw

        return result

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization.

        Removes cv2 and cv3 (one2many heads), keeping only one2one_cv2 and
        one2one_cv3 for NMS-free inference.
        """
        self.cv2 = None
        self.cv3 = None

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode polygon predictions for inference.

        Applies Sigmoid to xy offsets and Softplus to ray distances.
        Skips DFL entirely. Rays are scaled to pixel space using the
        training crop size (derived from feature map geometry), matching
        the normalisation used by RayCastTileDataset.
        """
        shape = x['feats'][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x['feats'], self.stride, 0.5))
            self.shape = shape

        poly = x['boxes']  # [B, raycast_dim, N_anchors]
        xy_offset = poly[:, :2, :].sigmoid()  # bounded [0, 1]
        rays = F.softplus(poly[:, 2:, :])  # strictly positive

        # Decode xy from grid-relative to absolute pixel coords
        xy_abs = (xy_offset * 2.0 - 0.5 + self.anchors) * self.strides

        # Rays: scale to pixel space using training crop size.
        # During training, rays are normalised by crop_size (from DataLoader),
        # so softplus outputs are in [0, ~1]. Multiply by imgsz to get pixels.
        imgsz = torch.tensor(shape[2:], device=poly.device, dtype=poly.dtype) * self.stride[0]
        rays_abs = rays * imgsz[0]  # [B, n_rays, N] — pixel-space ray distances

        dbox = torch.cat([xy_abs, rays_abs], dim=1)
        scores = x['scores'].sigmoid()
        return torch.cat((dbox, scores), 1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process end-to-end polygon predictions with top-k selection.

        Overrides Detect.postprocess() to handle raycast_dim polygon vectors
        instead of 4-dim bboxes. Called by Detect.forward() when end2end=True.

        Args:
            preds: [B, N_anchors, raycast_dim+nc] — decoded polygon + class scores.

        Returns:
            [B, max_det, raycast_dim+2] — top-k predictions with format
            [cx, cy, d_1..d_n, max_score, class_idx] per detection.
        """
        poly, scores = preds.split([self.raycast_dim, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        poly = poly.gather(dim=1, index=idx.repeat(1, 1, self.raycast_dim))
        return torch.cat([poly, scores, conf], dim=-1)

    def bias_init(self, crop_size: int = 256):
        """Initialize polygon head biases.

        XY channels (0-1): bias=0 for maximum sigmoid gradient (0.25) at init.
            sigmoid(0)=0.5 → offset=0.5 → decoded=(anchor+0.5)*stride/imgsz ≈ anchor_norm.
            This maximises gradient flow through the sigmoid bottleneck.
        Ray channels (2..2+n_rays): calibrated for ~15px radius cells at 0.25 MPP
            (lymphocytes 7-10μm → 28-40px diameter → radius ≈15px).
            During training, GT rays are normalized by crop_size, so
            softplus(bias) must equal target_px / crop_size.
        Centerness channel: bias=2.0 → sigmoid(2.0)≈0.88, starts permissive.

        Args:
            crop_size: Training crop size used for ray normalisation.
        """
        target_ray_px = 15.0  # reasonable lymphocyte radius at 0.25 MPP
        target_ray_norm = target_ray_px / crop_size
        ray_bias = math.log(math.exp(target_ray_norm) - 1)  # inverse softplus
        o2m = self.one2many
        for i, (a, b) in enumerate(zip(o2m['box_head'], o2m['cls_head'])):
            bias = a[-1].bias.data
            bias[:2] = 0.0
            bias[2:] = ray_bias  # rays: softplus → target_px / crop_size
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (crop_size / self.stride[i]) ** 2)
        if self._end2end_arg:
            o2o = self.one2one
            for i, (a, b) in enumerate(zip(o2o['box_head'], o2o['cls_head'])):
                bias = a[-1].bias.data
                bias[:2] = 0.0
                bias[2:] = ray_bias
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (crop_size / self.stride[i]) ** 2)

        if hasattr(self, 'aux_xy') and self.aux_xy is not None:
            for layer in self.aux_xy:
                layer.bias.data.zero_()
