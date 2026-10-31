"""Focused tests for LSP-DETR sliding-tile attention (STA) on decoder self-attention."""

from types import SimpleNamespace

import pytest
import torch

from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder, _unblock_empty_rows, sliding_tile_mask


def _reference_mask(n_side: int, kernel: int) -> torch.Tensor:
    """Direct port of LSP-DETR ``generate_sta_mask`` at tile==cell resolution."""
    kv_tiles = n_side
    out = torch.zeros(n_side * n_side, n_side * n_side, dtype=torch.bool)
    for q in range(n_side * n_side):
        qy, qx = divmod(q, n_side)
        cy = min(max(qy, kernel // 2), kv_tiles - 1 - kernel // 2)
        cx = min(max(qx, kernel // 2), kv_tiles - 1 - kernel // 2)
        for k in range(n_side * n_side):
            ky, kx = divmod(k, n_side)
            visible = abs(cy - ky) <= kernel // 2 and abs(cx - kx) <= kernel // 2
            out[q, k] = not visible
    return out


@pytest.mark.parametrize('n_side', [3, 5, 20])
@pytest.mark.parametrize('kernel', [1, 3, 5])
def test_mask_matches_lsp_reference(n_side, kernel):
    """The mask matches a brute-force port of LSP-DETR generate_sta_mask."""
    assert torch.equal(sliding_tile_mask(n_side, kernel), _reference_mask(n_side, kernel))


def test_mask_is_local_window():
    """An interior query sees exactly its kernel x kernel neighbourhood."""
    n_side, kernel = 20, 3
    mask = sliding_tile_mask(n_side, kernel)
    visible = ~mask
    assert visible.sum(-1).unique().tolist() == [kernel * kernel]
    # interior query (10, 10) indexes 10 * 20 + 10
    q = 10 * n_side + 10
    keys = visible[q].nonzero().flatten().tolist()
    expected = sorted(
        (y * n_side + x) for y in range(9, 12) for x in range(9, 12)
    )
    assert keys == expected


def test_mask_covers_diagonal_and_clamps_at_borders():
    """Mask covers diagonal and clamps at borders."""
    n_side = 7
    mask = sliding_tile_mask(n_side, 3)
    visible = ~mask
    assert not mask.diagonal().any()
    # Every query keeps a full 3x3 window: border queries clamp their centre
    # inward (LSP-DETR semantics) instead of shrinking the window.
    assert visible.sum(-1).unique().tolist() == [9]
    # The window is contiguous in the 2D tile index and never wraps rows.
    grid = (~mask).reshape(n_side, n_side, n_side, n_side)  # (qy, qx, ky, kx)
    for qy in range(n_side):
        for qx in range(n_side):
            ys = (grid[qy, qx].any(-1)).nonzero().flatten().tolist()
            xs = (grid[qy, qx].any(-2)).nonzero().flatten().tolist()
            assert ys == list(range(min(ys), max(ys) + 1))
            assert xs == list(range(min(xs), max(xs) + 1))
    # Clamping makes border-query windows shift inward, so the mask is not
    # symmetric; interior query (not on the border ring) is centred.
    q = 3 * n_side + 3
    assert (~mask)[q].nonzero().flatten().tolist() == sorted(
        2 * n_side + 2 + dy * n_side + dx for dy in range(3) for dx in range(3)
    )
    assert (mask != mask.t()).any()


def test_mask_rejects_bad_arguments():
    """Mask rejects bad arguments."""
    with pytest.raises(ValueError):
        sliding_tile_mask(0, 3)
    with pytest.raises(ValueError):
        sliding_tile_mask(4, 0)
    with pytest.raises(ValueError):
        sliding_tile_mask(4, 2)


def test_flag_requires_grid_layout():
    """Flag requires grid layout."""
    stub = SimpleNamespace(
        self_attention_sta=True,
        sta_kernel=3,
        embed=None,
    )
    # sta_n_side is only set by _grid_decoder_input
    stub_embed = torch.zeros(1, 25, 8)
    stub.embed = stub_embed
    stub.decoder = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(num_heads=1))])
    with pytest.raises(ValueError, match='grid query layout'):
        build = getattr(NuLiteDETRDecoder, '_sta_attention_mask')
        build(stub, stub_embed)


def test_sta_mask_blocks_non_window_queries_in_forward():
    """A DN-free grid forward with STA must block every out-of-window key."""
    n_side = 4
    stub = SimpleNamespace(
        self_attention_sta=True,
        sta_kernel=3,
        sta_n_side=n_side,
        decoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(num_heads=2))]),
    )
    embed = torch.zeros(2, n_side * n_side, 8)
    mask = NuLiteDETRDecoder._sta_attention_mask(stub, embed)
    assert mask.shape == (2 * 2, n_side * n_side, n_side * n_side)
    assert mask.dtype == torch.bool
    expected = sliding_tile_mask(n_side, 3)
    assert torch.equal(mask[0], expected)
    assert torch.equal(mask[-1], expected)


def test_sta_mask_composes_with_dn_isolation():
    """Sta mask composes with dn isolation."""
    n_side = 3
    n_dn = 2
    stub = SimpleNamespace(
        self_attention_sta=True,
        sta_kernel=3,
        sta_n_side=n_side,
        decoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(num_heads=1))]),
    )
    embed = torch.zeros(1, n_dn + n_side * n_side, 8)
    window = NuLiteDETRDecoder._sta_attention_mask(stub, embed)[0]
    # The STA mask only restricts the real-block window; DN isolation is a
    # separate mask composed in forward() (OR-ed), so DN rows stay open here.
    assert not window[:n_dn, :n_dn].any()
    assert not window[:n_dn, n_dn:].any()
    assert torch.equal(window[n_dn:, n_dn:], sliding_tile_mask(n_side, 3))
    # Composing with the DN isolation mask blocks DN<->real both ways.
    isolation = torch.zeros(n_dn + n_side * n_side, n_dn + n_side * n_side, dtype=torch.bool)
    isolation[:n_dn, n_dn:] = True
    isolation[n_dn:, :n_dn] = True
    combined = window | isolation
    assert combined[:n_dn, n_dn:].all()
    assert combined[n_dn:, :n_dn].all()
    assert torch.equal(combined[n_dn:, n_dn:], sliding_tile_mask(n_side, 3))


def test_window_composed_with_rank_causal_mask_never_leaves_empty_rows():
    """STA + MDS composition must not produce a fully blocked attention row."""
    n_side = 6
    nq = n_side * n_side
    window = sliding_tile_mask(n_side, 3)
    # Corner cell 0 is the global top-ranked query. Every local window that does
    # not contain it then has no strictly higher-ranked peer, so intersecting the
    # window with the rank-causal mask blocks every key (the NaN case).
    scores = torch.zeros(1, nq, 6)
    scores[0, 0, -1] = -10.0
    stub = SimpleNamespace(nhead=1)
    ranks = NuLiteDETRDecoder._rank_causal_mask(stub, scores)
    combined = window[None] | ranks
    assert combined.all(-1).any(), 'expected the composition to leave empty rows'
    rescued = _unblock_empty_rows(combined)
    assert not rescued.all(-1).any()
    # Rescued rows keep only their own diagonal; other rows are untouched.
    empty = combined.all(-1)
    assert rescued[empty].sum(-1).eq(1).all()
    assert torch.equal(rescued[~empty], combined[~empty])


def test_unblock_empty_rows_is_identity_when_no_row_is_blocked():
    """Unblock empty rows is identity when no row is blocked."""
    mask = sliding_tile_mask(5, 3)[None]
    assert torch.equal(_unblock_empty_rows(mask), mask)
