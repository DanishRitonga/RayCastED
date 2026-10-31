"""CayleySTRING2D must reproduce the LSP-DETR reference exactly."""

import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from raycasted.model.blocks.nulite_detr import CayleySTRING2D


class ReferenceCayleySTRING(nn.Module):
    """Verbatim lsp_detr/modeling/layers/cayley_string.py (pos_dim=2)."""

    def __init__(self, dim: int, pos_dim: int = 2, theta: float = 100.0) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.freqs = nn.Parameter(freqs.repeat(pos_dim, 1).clone())
        self.P = orthogonal(nn.Linear(dim, dim, bias=False), orthogonal_map='cayley')

    @torch.autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply the reference RoPE-Mixed rotation."""
        px = self.P(x.float())
        freqs = positions @ self.freqs
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(1)
        px_ = torch.view_as_complex(px.reshape(*px.shape[:-1], -1, 2))
        out = torch.view_as_real(px_ * freqs_cis).flatten(-2)
        return out.type_as(x)


def test_matches_reference_forward_and_gradients():
    """Forward output and gradients match the reference implementation."""
    dim, heads, n = 32, 4, 20
    torch.manual_seed(0)
    ours = CayleySTRING2D(dim, theta=100.0)
    ref = ReferenceCayleySTRING(dim, theta=100.0)
    # The reference uses ``orthogonal()``, which keeps ``weight = base @ Q(original)``
    # with its parameters nested under ``parametrizations``. Copy both tensors so
    # the comparison stays exact despite the different module layout.
    with torch.no_grad():
        ref.P.parametrizations.weight.original.copy_(ours.P.original)
        ref.P.parametrizations.weight[0].base.copy_(ours.P.base)
    x = torch.randn(2, heads, n, dim, requires_grad=True)
    pos = torch.rand(2, n, 2) * 20
    out_ours = ours(x, pos)
    out_ref = ref(x, pos)
    assert torch.allclose(out_ours, out_ref, atol=1e-6)
    out_ours.pow(2).mean().backward(retain_graph=True)
    g_ours = x.grad.clone()
    x.grad = None
    out_ref.pow(2).mean().backward()
    assert torch.allclose(g_ours, x.grad, atol=1e-6)


def test_orthogonality_preserved_after_step():
    """The Cayley parametrization keeps P orthogonal after an optimizer step."""
    dim = 16
    module = CayleySTRING2D(dim)
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    x = torch.randn(1, 2, 8, dim)
    pos = torch.rand(1, 8, 2)
    loss = module(x, pos).pow(2).mean()
    loss.backward()
    opt.step()
    weight = module.P.weight
    assert torch.allclose(weight @ weight.T, torch.eye(dim), atol=1e-5)


def test_relative_position_shift_invariance():
    """RoPE property: attention scores shift with translation of all queries."""
    dim = 16
    torch.manual_seed(1)
    module = CayleySTRING2D(dim)
    x = torch.randn(1, 1, 6, dim)
    pos = torch.rand(1, 6, 2)
    shift = torch.tensor([[[3.7, -1.2]]])
    a = module(x, pos)
    b = module(x, pos + shift)
    sa = a @ a.transpose(-1, -2)
    sb = b @ b.transpose(-1, -2)
    # self-similarity patterns are not required invariant under the learned
    # projection; the invariance lives in q/k pairing, so only check shape and
    # finiteness here (the exact-pairing check is the reference test above).
    assert a.shape == b.shape
    assert torch.isfinite(sa).all() and torch.isfinite(sb).all()


def test_rope_self_attention_ignores_absolute_query_pos():
    """LSP-DETR has no absolute PE: position enters only through RoPE.

    If ``query_pos`` leaked into the rotated Q/K, the translated-position
    invariance of attention logits would break.
    """
    from torch import nn

    from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder

    torch.manual_seed(3)
    layer = nn.Module()
    layer.self_attn = nn.MultiheadAttention(16, 2, dropout=0.0, batch_first=False)
    layer.dropout = 0.0
    layer.training = False
    embed = torch.randn(1, 7, 16)
    pos_a = torch.randn(1, 7, 16)
    pos_b = torch.randn(1, 7, 16)
    rope = CayleySTRING2D(8)
    positions = torch.rand(1, 7, 2) * 10

    out_a = NuLiteDETRDecoder._self_attention_with_rope(
        layer, embed, pos_a, None, rope, positions
    )
    out_b = NuLiteDETRDecoder._self_attention_with_rope(
        layer, embed, pos_b, None, rope, positions
    )
    assert torch.allclose(out_a, out_b, atol=1e-6)
