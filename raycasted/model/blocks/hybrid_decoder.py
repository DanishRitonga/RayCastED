"""Hybrid RayCast Decoder — LSP-DETR-style transformer with grid-based queries.

Replaces the FCN anchor-based head with a learned transformer decoder. Handles
feature fusion via cross-attention across multiple backbone scales. Hungarian
matching provides one-to-one assignment for NMS-free inference.

Architecture:
    YOLO26 backbone (layers 0-8) → [P2, P3, P4] features
        ↓
    FeatureSampling: project to hidden_dim, sample for query init
        ↓
    6× DecoderLayer: self-attn + cross-attn + SwiGLU FFN
        ↓
    Output: class logits, normalized centroid, log-space ray distances
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from raycasted.data.etl.utils import constants as _const

# ---------------------------------------------------------------------------
#  RoPE
# ---------------------------------------------------------------------------


class SinusoidalRoPE(nn.Module):
    """Standard 2D sinusoidal rotary position embedding.

    No learnable parameters — uses fixed sine/cosine frequencies per dimension.
    Applied separately to x and y coordinates, then concatenated.
    """

    def __init__(self, head_dim: int = 64):
        super().__init__()
        assert head_dim % 4 == 0, f'head_dim ({head_dim}) must be divisible by 4 for 2D RoPE'
        self.head_dim = head_dim
        self._half_dim = head_dim // 4

    def forward(self, q: torch.Tensor, k: torch.Tensor, q_coords: torch.Tensor, k_coords: torch.Tensor):
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor [B, H, Nq, D].
            k: Key tensor [B, H, Nk, D].
            q_coords: Query positions [B, Nq, 2] in pixel coordinates.
            k_coords: Key positions [B, Nk, 2] in pixel coordinates.
        """
        cos_q, sin_q = self._compute_freqs(q_coords, q)
        cos_k, sin_k = self._compute_freqs(k_coords, k)
        q_rotated = self._rotate_half(q, cos_q, sin_q)
        k_rotated = self._rotate_half(k, cos_k, sin_k)
        return q_rotated, k_rotated

    def _compute_freqs(self, coords: torch.Tensor, ref: torch.Tensor):
        """Compute cos/sin tables for coordinates.

        Args:
            coords: [B, N, 2] pixel coordinates.
            ref: Reference tensor for dtype/device.
        """
        device, dtype = ref.device, ref.dtype
        B, N = coords.shape[:2]
        freqs = torch.exp(torch.linspace(0, math.log(100.0), self._half_dim, device=device, dtype=dtype))
        coords_expanded = coords.unsqueeze(-1) * freqs  # [B, N, 2, half_dim]
        cos = coords_expanded.cos().view(B, N, -1)  # [B, N, 2*half_dim]
        sin = coords_expanded.sin().view(B, N, -1)
        cos = torch.cat([cos, cos], dim=-1)  # [B, N, D] — repeat for x,y pairs
        sin = torch.cat([sin, sin], dim=-1)
        return cos.unsqueeze(1), sin.unsqueeze(1)  # [B, 1, N, D]

    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * cos[..., : x1.shape[-1]] - x2 * sin[..., : x1.shape[-1]]
        y2 = x2 * cos[..., : x1.shape[-1]] + x1 * sin[..., : x1.shape[-1]]
        return torch.cat([y1, y2], dim=-1)


# ---------------------------------------------------------------------------
#  Feature projection & query initialization
# ---------------------------------------------------------------------------


class FeatureSampling(nn.Module):
    """Project backbone features to a common dimension and initialise queries.

    Each backbone level is projected via Conv2d(1×1) + LayerNorm.
    Query tokens are initialised by bilinear-sampling projected features at
    the grid query positions.
    """

    def __init__(self, feat_channels: list[int], hidden_dim: int = 384):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1, bias=False),
                nn.GroupNorm(1, hidden_dim),
            )
            for c in feat_channels
        )
        self.hidden_dim = hidden_dim

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Project each feature map level.

        Args:
            features: [P2, P3, P4] tensors, each [B, C_i, H_i, W_i].

        Returns:
            List of projected tensors, each [B, hidden_dim, H_i, W_i].
        """
        return [proj(f) for proj, f in zip(self.projections, features)]

    def init_queries(
        self,
        proj_features: list[torch.Tensor],
        query_positions: torch.Tensor,
        neck_idx: int = -1,
    ) -> torch.Tensor:
        """Bilinear-sample projected features at query positions.

        Uses the highest-level (smallest) feature map by default for stable
        query initialization.

        Args:
            proj_features: Projected features [P2, P3, P4] each [B, D, H, W].
            query_positions: [B, Q, 2] pixel coordinates for each query.
            neck_idx: Which feature level to sample from (default: last = P4).

        Returns:
            [B, Q, D] initial query embeddings.
        """
        feat = proj_features[neck_idx]
        B, D, H, W = feat.shape
        grid = query_positions.to(device=feat.device, dtype=feat.dtype).clone()
        grid[:, :, 0] = 2.0 * (grid[:, :, 0] + 0.5) / W - 1.0
        grid[:, :, 1] = 2.0 * (grid[:, :, 1] + 0.5) / H - 1.0
        grid = grid.unsqueeze(2)  # [B, Q, 1, 2]
        query_embeds = F.grid_sample(feat, grid, mode='bilinear', align_corners=False)
        return query_embeds.squeeze(2).transpose(1, 2)  # [B, Q, D]


# ---------------------------------------------------------------------------
#  Feed-forward network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (Llama4-style)."""

    def __init__(self, dim: int, ffn_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
#  Output heads
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Simple MLP with GELU activations."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            out_d = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
#  Decoder layer
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    """Transformer decoder layer: self-attn + cross-attn + FFN.

    Each layer: self_attn(tgt, tgt) → cross_attn(tgt, src) → FFN(tgt)
    with residual connections and layer-norm.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        head_dim: int = 64,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope = SinusoidalRoPE(head_dim)

        self.self_attn_q = nn.Linear(d_model, d_model, bias=False)
        self.self_attn_kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.self_attn_o = nn.Linear(d_model, d_model, bias=False)

        self.cross_attn_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_attn_kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.cross_attn_o = nn.Linear(d_model, d_model, bias=False)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def _reshape_for_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def _self_attention(
        self,
        tgt: torch.Tensor,
        q_coords: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Nq, _ = tgt.shape
        q = self._reshape_for_attn(self.self_attn_q(tgt))
        kv = self.self_attn_kv(tgt)
        k = self._reshape_for_attn(kv[..., : self.d_model])
        v = self._reshape_for_attn(kv[..., self.d_model :])
        q, k = self.rope(q, k, q_coords, q_coords)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Nq, self.d_model)
        return self.self_attn_o(out)

    def _cross_attention(
        self,
        tgt: torch.Tensor,
        src: torch.Tensor,
        q_coords: torch.Tensor,
        k_coords: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Nq, _ = tgt.shape
        B, Nk, _ = src.shape
        q = self._reshape_for_attn(self.cross_attn_q(tgt))
        kv = self.cross_attn_kv(src)
        k = self._reshape_for_attn(kv[..., : self.d_model])
        v = self._reshape_for_attn(kv[..., self.d_model :])
        q, k = self.rope(q, k, q_coords, k_coords)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Nq, self.d_model)
        return self.cross_attn_o(out)

    def forward(
        self,
        tgt: torch.Tensor,
        src: torch.Tensor,
        q_coords: torch.Tensor,
        k_coords: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.self_attn_norm(tgt + self._self_attention(tgt, q_coords, self_attn_mask))
        x = self.cross_attn_norm(x + self._cross_attention(x, src, q_coords, k_coords, cross_attn_mask))
        return self.ffn_norm(x + self.ffn(x))


# ---------------------------------------------------------------------------
#  Full decoder
# ---------------------------------------------------------------------------


class HybridRayCastDecoder(nn.Module):
    """LSP-DETR-style transformer decoder for ray-cast polygon detection.

    Takes multi-scale backbone features, applies feature projection, initialises
    a grid of learned queries, and iteratively refines them through 6 decoder
    layers with cross-attention to feature maps.

    Args:
        feat_channels: Input channel dimensions [C_P2, C_P3, C_P4].
        nc: Number of object classes.
        n_rays: Number of radial rays for polygon parameterisation (default 64).
        hidden_dim: Transformer hidden dimension (default 384).
        n_heads: Number of attention heads (default 6).
        ffn_dim: FFN hidden dimension (default 1024).
        num_layers: Number of decoder layers (default 6).
        query_block_size: Pixel spacing between grid queries (default 15).
        crop_size: Input image size for normalisation (default 256).
        dropout: Attention dropout (default 0.0).
        feature_levels: Per-layer feature level indices [2,1,0,2,1,0].
    """

    def __init__(
        self,
        feat_channels: tuple[int, ...] = (256, 512, 512),
        nc: int = 5,
        n_rays: int | None = None,
        hidden_dim: int = 384,
        n_heads: int = 6,
        ffn_dim: int = 1024,
        num_layers: int = 6,
        query_block_size: int = 15,
        crop_size: int = 256,
        dropout: float = 0.0,
        feature_levels: tuple[int, ...] = (2, 1, 0, 2, 1, 0),
    ):
        super().__init__()
        self.n_rays = n_rays if n_rays is not None else _const.N_RAYS
        self.nc = nc
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.num_layers = num_layers
        self.query_block_size = query_block_size
        self.crop_size = crop_size
        self.feature_levels = feature_levels
        self.raycast_dim = 2 + self.n_rays

        assert len(feature_levels) == num_layers, (
            f'feature_levels length ({len(feature_levels)}) must equal num_layers ({num_layers})'
        )

        self.feature_sampling = FeatureSampling(list(feat_channels), hidden_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=hidden_dim,
                    n_heads=n_heads,
                    head_dim=self.head_dim,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.class_head = nn.Linear(hidden_dim, nc + 1)
        self.point_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 2, 3) for _ in range(num_layers)])
        self.radial_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, self.n_rays, 3) for _ in range(num_layers)])

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.class_head.bias, math.log((1 - 0.01) / 0.01))

        for head_layer in self.point_head:
            last_linear = (
                head_layer.layers[-2] if isinstance(head_layer.layers[-2], nn.Linear) else head_layer.layers[-1]
            )
            if hasattr(last_linear, 'weight') and last_linear.weight is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

        for head_layer in self.radial_head:
            last_linear = (
                head_layer.layers[-2] if isinstance(head_layer.layers[-2], nn.Linear) else head_layer.layers[-1]
            )
            if hasattr(last_linear, 'weight') and last_linear.weight is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

    @staticmethod
    def _relative_to_absolute(coords: torch.Tensor, step: int) -> torch.Tensor:
        """Convert normalised coords [0,1] → absolute pixel coords."""
        coords = coords.sigmoid()  # [0, 1]
        device, dtype = coords.device, coords.dtype
        (B, H, W) = coords.shape if coords.dim() == 3 else coords.shape[:3]
        anchor_x = torch.arange(W, device=device, dtype=dtype) * step
        anchor_y = torch.arange(H, device=device, dtype=dtype) * step
        abs_x = coords[..., 0] * step + anchor_x.view(1, 1, W)
        abs_y = coords[..., 1] * step + anchor_y.view(1, H, 1)
        return torch.stack([abs_x, abs_y], dim=-1)

    @staticmethod
    def _build_feature_coords(
        features: list[torch.Tensor],
        crop_size: int,
    ) -> list[torch.Tensor]:
        """Build pixel coordinate grids for each feature map level."""
        coords_list = []
        for feat in features:
            B, _, H, W = feat.shape
            device = feat.device
            step_y = crop_size / H
            step_x = crop_size / W
            y_coords = torch.arange(H, device=device, dtype=feat.dtype) * step_y + step_y / 2.0
            x_coords = torch.arange(W, device=device, dtype=feat.dtype) * step_x + step_x / 2.0
            gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack([gx, gy], dim=-1)  # [H, W, 2]
            coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
            coords = coords.reshape(B, H * W, 2)
            coords_list.append(coords)
        return coords_list

    def _build_query_grid(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, int, int]:
        """Build 2D grid of query positions.

        Returns:
            query_positions: [B, Q, 2] pixel coordinates.
            grid_h, grid_w: Grid dimensions.
        """
        grid_w = int(math.ceil(self.crop_size / self.query_block_size))
        grid_h = int(math.ceil(self.crop_size / self.query_block_size))
        queries_per_dim = max(grid_w, grid_h)
        grid_w = grid_h = queries_per_dim
        num_queries = grid_h * grid_w

        anchor_x = torch.arange(grid_w, device=device, dtype=dtype) * self.query_block_size
        anchor_y = torch.arange(grid_h, device=device, dtype=dtype) * self.query_block_size
        gy, gx = torch.meshgrid(anchor_y, anchor_x, indexing='ij')
        positions = torch.stack([gx, gy], dim=-1)  # [H, W, 2]
        positions = positions.reshape(1, num_queries, 2).expand(batch_size, -1, -1)
        return positions, grid_h, grid_w

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the full decoder forward pass.

        Args:
            features: [P2, P3, P4] backbone features, each [B, C_i, H_i, W_i].

        Returns:
            Dict with keys:
                pred_logits: [B, Q, nc+1] class logits.
                pred_points: [B, Q, 2] normalised centroid positions [0,1].
                pred_radial: [B, Q, n_rays] log-space ray distances.
                aux_outputs: List of 5 intermediate dicts (same keys).
        """
        B = features[0].shape[0]
        device = features[0].device
        dtype = features[0].dtype

        proj_features = self.feature_sampling(features)
        query_pos, grid_h, grid_w = self._build_query_grid(B, device, dtype)
        num_queries = query_pos.shape[1]

        feature_coords = self._build_feature_coords(features, self.crop_size)

        tgt = self.feature_sampling.init_queries(proj_features, query_pos, neck_idx=-1)

        ref_points = torch.zeros(B, grid_h, grid_w, 2, device=device, dtype=dtype)
        radial_distances = torch.full(
            (B, num_queries, self.n_rays),
            math.log(self.query_block_size / 2),
            device=device,
            dtype=dtype,
        )

        new_ref_points = ref_points.clone().reshape(B, num_queries, 2)
        new_radial_distances = radial_distances.clone()

        aux_outputs = []

        for i, layer in enumerate(self.layers):
            level_idx = self.feature_levels[i]
            src = proj_features[level_idx]
            src_flat = src.flatten(2).transpose(1, 2)
            k_coords = feature_coords[level_idx]

            tgt = tgt.reshape(B, num_queries, self.hidden_dim)
            tgt = layer(tgt, src_flat, query_pos, k_coords)

            delta_point = self.point_head[i](tgt)
            delta_radial = self.radial_head[i](tgt)

            cur_ref = new_ref_points + delta_point
            cur_radial = new_radial_distances + delta_radial

            abs_points = self._relative_to_absolute(
                cur_ref.reshape(B, grid_h, grid_w, 2), self.query_block_size
            ).reshape(B, num_queries, 2)

            if self.training:
                logits = self.class_head(tgt)
                aux_outputs.append(
                    {
                        'pred_logits': logits,
                        'pred_points': abs_points / self.crop_size,
                        'pred_radial': cur_radial,
                    }
                )

            new_ref_points = ref_points.reshape(B, num_queries, 2) + delta_point
            new_radial_distances = radial_distances + delta_radial

            query_pos = abs_points.detach()

            if self.training:
                ref_points = new_ref_points.detach().reshape(B, grid_h, grid_w, 2)
                radial_distances = new_radial_distances.detach()
                tgt = tgt.detach()

        if not self.training:
            return {
                'pred_logits': self.class_head(tgt),
                'pred_points': abs_points / self.crop_size,
                'pred_radial': cur_radial,
                'aux_outputs': [],
            }

        return {
            'pred_logits': aux_outputs[-1]['pred_logits'],
            'pred_points': aux_outputs[-1]['pred_points'],
            'pred_radial': aux_outputs[-1]['pred_radial'],
            'aux_outputs': aux_outputs[:-1],
        }
