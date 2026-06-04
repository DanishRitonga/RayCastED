"""GAT-based GNN decoder layers replacing transformer self/cross-attention.

QuerySelfGNN: k-NN graph on query positions → GAT message passing.
QueryCrossGNN: each query attends to k-nearest feature map tokens.
GNNDecoderLayer: drop-in replacement for Layer/DecoderLayer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .lsp_detr_arch import FeedForward


def _batched_index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather values along dim using 3D index tensor.

    Args:
        input: [B, N, D]
        dim: dimension to index along (typically 1)
        index: [B, Q, K] — indices into dim

    Returns:
        [B, Q, K, D]
    """
    B, N, D = input.shape
    _, Q, K = index.shape
    flat_input = input.reshape(B * N, D)
    batch_offset = torch.arange(B, device=index.device, dtype=index.dtype).reshape(B, 1, 1) * N
    flat_index = (index + batch_offset).reshape(-1)
    return flat_input[flat_index].reshape(B, Q, K, D)


class QuerySelfGNN(nn.Module):
    """GAT self-attention: queries attend to k spatial neighbors."""

    def __init__(self, dim: int, num_heads: int = 8, k: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k = k
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_edge_proj = nn.Linear(dim + 4, dim * 2)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        B, H, W, D = x.shape
        Q = H * W
        x_flat = x.reshape(B, Q, D)

        with torch.no_grad():
            dist = torch.cdist(coords, coords)
            _, knn_idx = torch.topk(dist, self.k + 1, dim=-1, largest=False)
            knn_idx = knn_idx[..., 1:]

        neighbor_x = _batched_index_select(x_flat, 1, knn_idx)
        neighbor_coords = _batched_index_select(coords, 1, knn_idx)

        rel_pos = neighbor_coords - coords.unsqueeze(2)
        dist_e = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_feat = torch.cat([rel_pos, dist_e, 1.0 / (dist_e + 1e-6)], dim=-1)

        q = self.q_proj(x_flat).view(B, Q, 1, self.num_heads, self.head_dim)

        neighbor_with_edge = torch.cat([neighbor_x, edge_feat], dim=-1)
        kv = self.kv_edge_proj(neighbor_with_edge)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, Q, self.k, self.num_heads, self.head_dim)
        v = v.view(B, Q, self.k, self.num_heads, self.head_dim)

        attn = (q * k).sum(dim=-1) * self.scale
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * v).sum(dim=2)
        out = out.view(B, Q, D)
        out = self.o_proj(out)
        return self.dropout(out).reshape(B, H, W, D)


class QueryCrossGNN(nn.Module):
    """GAT cross-attention: each query attends to k-nearest feature map tokens."""

    def __init__(self, dim: int, src_dim: int, num_heads: int = 8, k: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k = k
        self.scale = self.head_dim**-0.5

        self.src_proj = nn.Linear(src_dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_edge_proj = nn.Linear(dim + 4, dim * 2)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, tgt_coords: Tensor, src: Tensor, src_coords: Tensor) -> Tensor:
        B, Hq, Wq, D = tgt.shape
        Q = Hq * Wq
        Hs, Ws, Ds = src.shape[1], src.shape[2], src.shape[3]
        N_kv = Hs * Ws

        tgt_flat = tgt.reshape(B, Q, D)
        src_flat = src.reshape(B, N_kv, Ds)

        with torch.no_grad():
            dist = torch.cdist(tgt_coords, src_coords)
            _, knn_idx = torch.topk(dist, self.k, dim=-1, largest=False)

        neighbor_src = _batched_index_select(src_flat, 1, knn_idx)
        neighbor_coords = _batched_index_select(src_coords, 1, knn_idx)

        rel_pos = neighbor_coords - tgt_coords.unsqueeze(2)
        dist_e = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_feat = torch.cat([rel_pos, dist_e, 1.0 / (dist_e + 1e-6)], dim=-1)

        neighbor_proj = self.src_proj(neighbor_src)
        neighbor_with_edge = torch.cat([neighbor_proj, edge_feat], dim=-1)

        q = self.q_proj(tgt_flat).view(B, Q, 1, self.num_heads, self.head_dim)

        kv = self.kv_edge_proj(neighbor_with_edge)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, Q, self.k, self.num_heads, self.head_dim)
        v = v.view(B, Q, self.k, self.num_heads, self.head_dim)

        attn = (q * k).sum(dim=-1) * self.scale
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * v).sum(dim=2)
        out = out.view(B, Q, D)
        out = self.o_proj(out)
        return self.dropout(out).reshape(B, Hq, Wq, D)


class GNNDecoderLayer(nn.Module):
    """Drop-in replacement for Layer/DecoderLayer using GNN attention."""

    def __init__(
        self,
        dim: int,
        src_dim: int,
        num_heads: int,
        self_k: int = 8,
        cross_k: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_gnn = QuerySelfGNN(dim, num_heads, self_k, dropout)
        self.self_norm = nn.LayerNorm(dim)
        self.cross_gnn = QueryCrossGNN(dim, src_dim, num_heads, cross_k, dropout)
        self.cross_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 4)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, tgt: Tensor, src: Tensor, tgt_coords: Tensor, src_coords: Tensor) -> Tensor:
        x = self.self_gnn(tgt, tgt_coords)
        tgt = self.self_norm(tgt + x)
        x = self.cross_gnn(tgt, tgt_coords, src, src_coords)
        tgt = self.cross_norm(tgt + x)
        return self.ffn_norm(tgt + self.ffn(tgt))
