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
from raycasted.model.blocks.dcn import DCNConv

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


class PredictionRefinementAttention(nn.Module):
    """Self-attention on top-K scored predictions for duplicate suppression.

    After the FCN scores all 5376 anchors, the top-K (e.g., 100) by
    confidence are selected. These K predictions (mostly fg) undergo
    standard O(N^2) multi-head self-attention, letting each prediction
    "see" all others. A final linear head outputs a 1-channel suppression
    logit per prediction — at inference: ``final_conf = cls * sigmoid(refine)``.

    This is fundamentally different from AnchorSelfAttention (train31/32)
    which applied linear attention on ALL 5376 anchors (98.7% bg) at the
    feature level, washing out fg/bg discrimination. Here, the 5376→K
    selection happens BEFORE interaction, so attention operates on a
    predominantly-fg set.

    Inspired by Efficient DETR (dense→sparse), Sparse R-CNN (proposal
    self-attention), and Relation Network (inter-proposal attention).

    Args:
        feat_dim: Input feature dimension per prediction (c3 from cls head).
        num_heads: Number of attention heads.
        ff_dim: Feed-forward intermediate dimension.
        dropout: Dropout rate.
    """

    def __init__(self, feat_dim: int, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.suppress_head = nn.Linear(feat_dim, 1)
        self._init_suppress_head()

    def _init_suppress_head(self):
        nn.init.zeros_(self.suppress_head.weight)
        nn.init.constant_(self.suppress_head.bias, 2.0)

    @staticmethod
    def build_2d_sincos_pe(w: int, h: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 4 == 0, f'embed_dim {embed_dim} must be divisible by 4'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (10000.0**omega)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        gw, gh = torch.meshgrid(grid_w, grid_h, indexing='ij')
        out_w = gw.flatten()[..., None] @ omega[None]
        out_h = gh.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply self-attention on top-K prediction tokens.

        Args:
            tokens: [B, K, feat_dim] — per-prediction features (c3 cls features).
            positions: [B, K, 2] — normalised (x, y) centroid positions in [0, 1].

        Returns:
            [B, K, 1] — suppression logits. Sigmoid gives multiplicative weight.
        """
        B, K, C = tokens.shape
        pe_embed_dim = C // 4 * 4
        if pe_embed_dim < C:
            pe_embed_dim = max(4, (C // 4) * 4)
        pe_embed_dim = min(pe_embed_dim, C)
        if pe_embed_dim % 4 != 0:
            pe_embed_dim = ((pe_embed_dim // 4) * 4) or 4

        pe = self.build_2d_sincos_pe(K, 1, pe_embed_dim).to(device=tokens.device, dtype=tokens.dtype)
        if pe_embed_dim < C:
            pe_padded = torch.zeros(1, K, C, device=tokens.device, dtype=tokens.dtype)
            pe_padded[:, :, :pe_embed_dim] = pe.unsqueeze(0)
            pe = pe_padded.squeeze(0)
        tokens = tokens + pe[:K, :C].unsqueeze(0)

        out = self.encoder_layer(tokens)
        suppress_logits = self.suppress_head(out)
        return suppress_logits


class RayCastDetect(Detect):
    """Polygon detection head replacing bounding-box regression with raycast.

    Subclasses ultralytics Detect, replacing the cv2 regression branch with:
        Conv → Conv → RayRefinementBlock → Conv2d(c2, raycast_dim, 1)

    DFL is removed entirely. Output activations:
        - Channels 0-1 (xy): Sigmoid (inference only)
        - Channels 2..(2+n_rays) (rays): Softplus (inference only)
    """

    max_det = 100

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        n_rays: int | None = None,
        head_channel_scale: float = 0.5,
        head_channel_min: int = 64,
        cls_channel_scale: float = 1.0,
        cls_channel_min: int = 0,
        refinement_kernel_size: int = 3,
        aux_xy: bool = False,
        prediction_refinement: bool = False,
        prediction_refinement_topk: int = 100,
        inter_scale_competition: bool = False,
        inter_scale_temperature: float = 1.0,
        local_competition: bool = False,
        local_competition_kernel: int = 3,
        local_competition_temperature: float = 1.0,
        dcn_in_reg_head: bool = False,
        dcn_in_cls_head: bool = False,
        hierarchical_cls: bool = False,
    ):
        """Initialize polygon detection head.

        Args:
            nc: Number of classes.
            reg_max: DFL channels (ignored for polygon head, kept for API compat).
            end2end: Whether to use end-to-end NMS-free detection.
            ch: Tuple of channel sizes from backbone feature maps.
            n_rays: Number of radial rays for polygon parameterization.
            head_channel_scale: Fraction of input channels for regression head width.
            head_channel_min: Minimum regression head intermediate channels.
            cls_channel_scale: Scale factor for cls head intermediate channels.
                c3_new = max(cls_channel_min, int(ch[0] * cls_channel_scale)).
                Default 1.0 preserves original c3 = max(ch[0], min(nc, 100)).
            cls_channel_min: Minimum cls head intermediate channels.
                0 = use ch[0] as floor (same as original formula).
            refinement_kernel_size: Kernel size for polygon refinement block.
                3 = standard RayRefinementBlock (default).
                7 or 13 = LargeKernelRefinementBlock (LKCell-style, wider receptive field).
            aux_xy: If True, attach a lightweight 1x1 conv head for auxiliary xy regression
                directly on neck features, bypassing the 4-layer head stack.
            prediction_refinement: If True, apply self-attention on top-K scored
                predictions AFTER the FCN head. Each prediction sees all other top-K
                predictions, enabling inter-prediction competition (learned NMS).
                Fundamentally different from feature-level attention (train31/32/34)
                which operated on 5376 bg-dominated anchor features.
            prediction_refinement_topk: Number of top predictions to refine (default 100).
            inter_scale_competition: If True, softmax competition across scales at inference.
                Upsamples P3/P4 cls to P2 resolution, stacks, softmax across scale dim,
                multiplies each scale's confidence by its competition weight. Suppresses
                cross-scale duplicates (same nucleus predicted at P2 AND P3).
            inter_scale_temperature: Softmax temperature for inter-scale competition.
                Lower = sharper (winner-take-more). 1.0 = standard softmax.
            local_competition: If True, per-scale 3×3 neighborhood softmax at inference.
                Each anchor competes with its 8 neighbors — local winner-take-more.
                Suppresses same-scale duplicates where one nucleus fires adjacent anchors.
            local_competition_kernel: Neighborhood size for local competition (default 3 = 3×3).
            local_competition_temperature: Softmax temperature for local competition (lower = sharper).
            dcn_in_reg_head: If True, replace 2nd Conv in cv2 with DCNConv (modulated
                deformable conv). Zero-init offsets/mask so it starts as standard conv.
            dcn_in_cls_head: If True, replace 2nd Conv in cv3 with DCNConv.
            hierarchical_cls: If True, split o2o cls head into binary (fg/bg, 1ch)
                + class (cell type, nc ch). Binary head gets ALL 5376 anchors of gradient
                for strong fg/bg discrimination. Class head only learns inter-class
                separation on fg anchors. At inference: sigmoid(binary) × softmax(class).
        """
        self.n_rays = n_rays if n_rays is not None else _const.N_RAYS
        self.raycast_dim = 2 + self.n_rays  # xy + rays
        self._end2end_arg = end2end  # store before parent __init__ (end2end is a property)
        self._prediction_refinement = prediction_refinement
        self.prediction_refinement_topk = prediction_refinement_topk
        self.inter_scale_competition = inter_scale_competition
        self.inter_scale_temperature = inter_scale_temperature
        self.local_competition = local_competition
        self.local_competition_kernel = local_competition_kernel
        self.local_competition_temperature = local_competition_temperature
        self.hierarchical_cls = hierarchical_cls

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
        if dcn_in_reg_head:
            self.cv2 = nn.ModuleList(
                nn.Sequential(
                    Conv(x, c2, 3),
                    DCNConv(c2, c2, 3),
                    block_cls(c2, **block_kwargs),
                    nn.Conv2d(c2, self.raycast_dim, 1),
                )
                for x in ch
            )
        else:
            self.cv2 = nn.ModuleList(
                nn.Sequential(
                    Conv(x, c2, 3),
                    Conv(c2, c2, 3),
                    block_cls(c2, **block_kwargs),
                    nn.Conv2d(c2, self.raycast_dim, 1),
                )
                for x in ch
            )

        # Replace cv3 (cls head) with scaled intermediate channels
        # Parent Detect.__init__ builds cv3 with c3 = max(ch[0], min(nc, 100)).
        # We rebuild with c3_new = max(cls_channel_min, int(ch[0] * cls_channel_scale))
        # to give the cls head more capacity for fg/bg discrimination.
        c3_original = max(ch[0], min(nc, 100))
        _scale_c3 = cls_channel_scale != 1.0 or cls_channel_min > 0
        c3 = max(cls_channel_min, int(ch[0] * cls_channel_scale)) if _scale_c3 else c3_original
        if c3 != c3_original or dcn_in_cls_head:
            from ultralytics.nn.modules.conv import DWConv

            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(
                        DWConv(c3, c3, 3),
                        DCNConv(c3, c3, 3) if dcn_in_cls_head else Conv(c3, c3, 1),
                    ),
                    nn.Conv2d(c3, self.nc, 1),
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

        if prediction_refinement:
            self.prediction_refinement_attn = PredictionRefinementAttention(
                feat_dim=c3,
                num_heads=4,
                ff_dim=256,
                dropout=0.0,
            )
        else:
            self.prediction_refinement_attn = None

        # Separate o2o heads — both box (cv2) and cls (cv3) are deepcopied.
        # Shared cv3 caused 11.5x overprediction: o2m's dense positives (topk=15)
        # taught the shared cls head to fire high scores for many anchors per GT,
        # defeating o2o's topk2=1 duplicate suppression.
        if self._end2end_arg:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # deepcopy scaled cv3 for o2o

            if hierarchical_cls:
                # Hierarchical cls: binary (fg/bg) + class (cell type) heads.
                # Binary head: ALL 5376 anchors get gradient → strong fg/bg signal.
                # Class head: only fg anchors → simpler inter-class task.
                self.one2one_cv3_binary = copy.deepcopy(self.cv3)
                for seq in self.one2one_cv3_binary:
                    seq[-1] = nn.Conv2d(seq[-1].in_channels, 1, 1)
                self.one2one_cv3_class = copy.deepcopy(self.cv3)
                # one2one_cv3 kept for o2m fallback / fuse compat; o2o uses binary+class

            if self.prediction_refinement_attn is not None:
                self.one2one_prediction_refinement_attn = copy.deepcopy(self.prediction_refinement_attn)
                self.prediction_refinement_attn = None

    @property
    def one2many(self):
        """Return one2many head components."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Return one2one head components — separate cls head for NMS-free inference."""
        result = dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)
        if self.hierarchical_cls:
            result['cls_head_binary'] = self.one2one_cv3_binary
            result['cls_head_class'] = self.one2one_cv3_class
        if hasattr(self, 'one2one_prediction_refinement_attn') and self.one2one_prediction_refinement_attn is not None:
            result['prediction_refinement_attn'] = self.one2one_prediction_refinement_attn
        return result

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
        prediction_refinement_attn: nn.Module | None = None,
        cls_head_binary: nn.Module | None = None,
        cls_head_class: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Concatenate polygon predictions and class scores across scales.

        Returns dict with 'boxes' key containing raycast_dim polygon logits,
        'scores' key containing class logits, and 'feats' key with feature maps.

        When cls_head_binary and cls_head_class are provided (hierarchical cls),
        returns 'binary_scores' [B, 1, N] and 'class_scores' [B, nc, N] instead
        of 'scores'.
        """
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        poly = torch.cat([box_head[i](x[i]).view(bs, self.raycast_dim, -1) for i in range(self.nl)], dim=-1)

        if cls_head_binary is not None and cls_head_class is not None:
            binary_scores = torch.cat([cls_head_binary[i](x[i]).view(bs, 1, -1) for i in range(self.nl)], dim=-1)
            class_scores = torch.cat([cls_head_class[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
            result = dict(boxes=poly, binary_scores=binary_scores, class_scores=class_scores, feats=x)
        else:
            scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
            result = dict(boxes=poly, scores=scores, feats=x)

        if hasattr(self, 'aux_xy') and self.aux_xy is not None:
            aux_raw = torch.cat([self.aux_xy[i](x[i]).view(bs, 2, -1) for i in range(self.nl)], dim=-1)
            result['aux_xy_raw'] = aux_raw

        if prediction_refinement_attn is not None:
            c3_feats = []
            for i in range(self.nl):
                feat = cls_head[i][:-1](x[i])  # [B, c3, H, W]
                h, w = feat.shape[2], feat.shape[3]
                c3_feats.append(feat.permute(0, 2, 3, 1).reshape(bs, h * w, -1))
            cls_feats_flat = torch.cat(c3_feats, dim=1)  # [B, N_total, c3]
            result['cls_feats_flat'] = cls_feats_flat

        return result

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization.

        Removes cv2 and cv3 (one2many heads), keeping only one2one_cv2
        and one2one_cv3 for NMS-free inference.
        """
        self.cv2 = None
        self.cv3 = None

    def _apply_prediction_refinement(self, one2one_preds: dict) -> dict:
        """Apply prediction-level self-attention on top-K scored predictions.

        Selects top-K predictions by max class confidence, applies
        TransformerEncoder self-attention among them, and outputs a
        per-prediction suppression weight (sigmoid → multiply with cls score).

        This is fundamentally different from feature-level attention
        (train31/32/34): it operates on K=100 mostly-fg predictions
        AFTER the FCN scores them, not on 5376 bg-dominated anchor features.

        Args:
            one2one_preds: Dict from forward_head with 'scores', 'cls_feats_flat',
                'boxes', 'feats' keys.

        Returns:
            Modified dict with 'refine_suppress' [B, nc, N] and
            'refine_raw' [B, K, 1] for loss computation.
        """
        if 'cls_feats_flat' not in one2one_preds:
            return one2one_preds
        attn = self.one2one_prediction_refinement_attn if hasattr(self, 'one2one_prediction_refinement_attn') else None
        if attn is None:
            attn = self.prediction_refinement_attn if hasattr(self, 'prediction_refinement_attn') else None
        if attn is None:
            return one2one_preds

        scores = one2one_preds['scores']  # [B, nc, N] — raw logits
        cls_feats = one2one_preds['cls_feats_flat']  # [B, N, c3]
        bs, nc, N = scores.shape
        K = self.prediction_refinement_topk

        max_scores = scores.sigmoid().max(dim=1).values  # [B, N]
        topk_vals, topk_idx = max_scores.topk(min(K, N), dim=1)  # [B, K]

        topk_feats = torch.gather(cls_feats, 1, topk_idx.unsqueeze(-1).expand(-1, -1, cls_feats.shape[-1]))

        # Compute normalised centroid positions for sincos PE
        feats = one2one_preds['feats']
        shape = feats[0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(feats, self.stride, 0.5))
            self.shape = shape
        anchors = self.anchors  # [2, N]
        strides = self.strides  # [1, N] or [2, N]
        if strides.dim() == 2 and strides.shape[0] == 2:
            strides = strides[:1, :]
        if strides.dim() == 3:
            strides = strides.squeeze(0)
        if anchors.dim() == 3:
            anchors = anchors.squeeze(0)

        xy_raw = one2one_preds['boxes'][:, :2, :].sigmoid()  # [B, 2, N]
        xy_px = (xy_raw * 2.0 - 0.5 + anchors.unsqueeze(0)) * strides.unsqueeze(0)
        imgsz = strides.max().item() * feats[0].shape[2]
        xy_norm = (xy_px / max(imgsz, 1)).clamp(0, 1)
        topk_xy = torch.gather(xy_norm, 2, topk_idx.unsqueeze(1).expand(-1, 2, -1))
        topk_pos = topk_xy.permute(0, 2, 1)

        suppress_logits = attn(topk_feats, topk_pos)  # [B, K, 1]
        suppress_weights = suppress_logits.sigmoid()  # [B, K, 1]

        full_suppress = torch.ones(bs, 1, N, device=scores.device, dtype=scores.dtype)
        topk_expand = topk_idx.unsqueeze(1).expand(-1, 1, -1)
        full_suppress.scatter_(2, topk_expand, suppress_weights.permute(0, 2, 1))

        one2one_preds['refine_suppress'] = full_suppress
        one2one_preds['refine_raw'] = suppress_logits
        one2one_preds['refine_topk_idx'] = topk_idx
        return one2one_preds

    def forward(self, x: list[torch.Tensor]):
        """Override Detect.forward to apply prediction-level refinement."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one_kwargs = dict(self.one2one)
            has_refine = (
                hasattr(self, 'one2one_prediction_refinement_attn')
                and self.one2one_prediction_refinement_attn is not None
            )
            if not has_refine:
                has_refine = hasattr(self, 'prediction_refinement_attn') and self.prediction_refinement_attn is not None
            if has_refine:
                one2one_kwargs.pop('prediction_refinement_attn', None)
            one2one = self.forward_head(x_detach, **one2one_kwargs)
            if has_refine:
                one2one = self._apply_prediction_refinement(one2one)
            preds = {'one2many': preds, 'one2one': one2one}
        if self.training:
            return preds
        y = self._inference(preds['one2one'] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _apply_local_competition(
        self,
        scores: torch.Tensor,
        feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """Per-scale 3x3 neighborhood softmax to suppress same-scale duplicates.

        For each scale's cls map, computes softmax over a k×k neighborhood.
        Anchors that are the local maximum get weight ~1.0; neighbors get ~0.
        This suppresses duplicate predictions where a single nucleus fires
        multiple adjacent anchors on the same scale.

        Args:
            scores: [B, nc, N_total] — sigmoid-activated cls scores.
            feats: List of feature maps per scale [B, C, H_i, W_i].

        Returns:
            [B, nc, N_total] — locally-competition-adjusted cls scores.
        """
        bs, nc, N_total = scores.shape
        spatial_shapes = [(f.shape[2], f.shape[3]) for f in feats]
        anchor_counts = [h * w for h, w in spatial_shapes]
        scale_scores = scores.split(anchor_counts, dim=-1)

        k = self.local_competition_kernel
        pad = k // 2

        result_parts = []
        for ss, (h, w) in zip(scale_scores, spatial_shapes):
            spatial = ss.view(bs, nc, h, w)
            unfolded = F.unfold(spatial, kernel_size=k, padding=pad)  # [B, nc*k*k, h*w]
            unfolded = unfolded.view(bs, nc, k * k, h, w)
            weights = F.softmax(unfolded / self.local_competition_temperature, dim=2)
            center_idx = k * k // 2
            out = spatial * weights[:, :, center_idx, :, :]
            result_parts.append(out.view(bs, nc, -1))

        return torch.cat(result_parts, dim=-1)

    def _apply_inter_scale_competition(
        self,
        scores: torch.Tensor,
        feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """Softmax competition across scales to suppress cross-scale duplicates.

        For each spatial position at P2 resolution, computes softmax across
        the scale dimension. Each scale's confidence is multiplied by its
        competition weight — the winning scale gets ~1.0, losers get ~0.

        This is differentiable (for future training use) and inference-only
        for now (no training loss changes).

        Args:
            scores: [B, nc, N_total] — sigmoid-activated cls scores (concatenated across scales).
            feats: List of feature maps per scale [B, C, H_i, W_i].

        Returns:
            [B, nc, N_total] — competition-adjusted cls scores.
        """
        bs, nc, N_total = scores.shape

        spatial_shapes = [(f.shape[2], f.shape[3]) for f in feats]
        anchor_counts = [h * w for h, w in spatial_shapes]
        H_p2, W_p2 = spatial_shapes[0]

        scale_scores = scores.split(anchor_counts, dim=-1)

        scale_spatial = []
        for si, (ss, (h, w)) in enumerate(zip(scale_scores, spatial_shapes)):
            scale_spatial.append(ss.view(bs, nc, h, w))

        upsampled = []
        for si, ss in enumerate(scale_spatial):
            if si == 0:
                upsampled.append(ss)
            else:
                upsampled.append(F.interpolate(ss, size=(H_p2, W_p2), mode='bilinear', align_corners=False))

        stacked = torch.stack(upsampled, dim=2)

        temp = self.inter_scale_temperature
        if temp != 1.0:
            stacked = stacked / temp

        competition_weights = F.softmax(stacked, dim=2)

        for si in range(len(scale_spatial)):
            cw = competition_weights[:, :, si, :, :]  # [B, nc, H_p2, W_p2]
            if si == 0:
                adjusted = scale_spatial[si] * cw
            else:
                cw_down = F.interpolate(cw, size=spatial_shapes[si], mode='bilinear', align_corners=False)
                adjusted = scale_spatial[si] * cw_down

            if si == 0:
                result_parts = [adjusted.view(bs, nc, -1)]
            else:
                result_parts.append(adjusted.view(bs, nc, -1))

        return torch.cat(result_parts, dim=-1)

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
        rays = F.softplus(poly[:, 2:, :])  # strictly positive, [B, n_rays, N]

        # Decode xy from grid-relative to absolute pixel coords
        xy_abs = (xy_offset * 2.0 - 0.5 + self.anchors) * self.strides

        # Rays: scale to pixel space using training crop size.
        # During training, rays are normalised by crop_size (from DataLoader),
        # so softplus outputs are in [0, ~1]. Multiply by imgsz to get pixels.
        imgsz = torch.tensor(shape[2:], device=poly.device, dtype=poly.dtype) * self.stride[0]
        rays_abs = rays * imgsz[0]  # [B, n_rays, N] — pixel-space ray distances

        dbox = torch.cat([xy_abs, rays_abs], dim=1)

        # Hierarchical cls: combine binary fg/bg + class distribution
        if 'binary_scores' in x and 'class_scores' in x:
            binary_prob = x['binary_scores'].sigmoid()  # [B, 1, N]
            class_prob = F.softmax(x['class_scores'], dim=1)  # [B, nc, N]
            scores = binary_prob * class_prob  # [B, nc, N]
        else:
            scores = x['scores'].sigmoid()

        # Local competition: per-scale 3×3 neighborhood softmax suppresses
        # same-scale duplicates (adjacent anchors firing for one nucleus).
        if self.local_competition and self.nl > 0:
            scores = self._apply_local_competition(scores, x['feats'])

        # Inter-scale competition: softmax across scales suppresses cross-scale duplicates.
        # Same nucleus often predicted at both P2 and P3 with high confidence.
        # Competition: upsample all scales to P2, softmax across scale dim,
        # multiply each scale's confidence by its competition weight.
        if self.inter_scale_competition and self.nl > 1:
            scores = self._apply_inter_scale_competition(
                scores,
                x['feats'],
            )

        if 'refine_suppress' in x:
            scores = scores * x['refine_suppress']
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
            if self.hierarchical_cls:
                for i in range(self.nl):
                    self.one2one_cv3_binary[i][-1].bias.data.fill_(math.log(5 / 1 / (crop_size / self.stride[i]) ** 2))
                    self.one2one_cv3_class[i][-1].bias.data[: self.nc] = math.log(
                        5 / self.nc / (crop_size / self.stride[i]) ** 2
                    )

        if hasattr(self, 'aux_xy') and self.aux_xy is not None:
            for layer in self.aux_xy:
                layer.bias.data.zero_()
