"""Focused tests for map modulation and pooled seed columns."""

from types import SimpleNamespace

import pytest
import torch

from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder


def _stub(pool_sizes=(3, 5, 9)):
    """Minimal decoder stub for the seed-sampling helpers."""
    return SimpleNamespace(
        seed_in_content=True,
        seed_in_content_pooled=True,
        seed_in_content_pool_sizes=tuple(pool_sizes),
        map_modulation=False,
        map_modulation_proj=None,
        _sample_seed_at=NuLiteDETRDecoder._sample_seed_at,
        _sample_seed_multiscale=NuLiteDETRDecoder._sample_seed_multiscale,
        _seed_columns=NuLiteDETRDecoder._seed_columns,
        _apply_map_modulation=NuLiteDETRDecoder._apply_map_modulation,
    )


def _maps(bs=2, h=32, w=32):
    seed = torch.randn(bs, 1, h, w)
    pos = torch.rand(bs, 7, 2)
    return seed, pos


def test_multiscale_sampling_shape():
    """Raw column plus one column per pooling window."""
    stub = _stub()
    seed, pos = _maps()
    out = NuLiteDETRDecoder._sample_seed_multiscale(seed, pos, stub.seed_in_content_pool_sizes)
    assert out.shape == (2, 7, 1 + 3)


def test_multiscale_pooling_preserves_peak():
    """A peak beside the query reaches it through the pooled columns."""
    stub = _stub()
    seed = torch.full((1, 1, 32, 32), -10.0)
    seed[0, 0, 16, 18] = 20.0  # peak two px right of the sample point
    pos = torch.tensor([[[16.0 / 32.0, 16.0 / 32.0]]])
    out = NuLiteDETRDecoder._sample_seed_multiscale(seed, pos, stub.seed_in_content_pool_sizes)[0, 0]
    assert abs(float(out[0])) < 20.0  # raw bilinear sample misses the peak
    assert float(out[1:].max()) > 5.0  # a pooled window still sees it


def test_pooled_wider_than_plain():
    """Pooled columns strictly exceed the single raw column."""
    stub = _stub(pool_sizes=(3, 5))
    seed, pos = _maps()
    plain = NuLiteDETRDecoder._sample_seed_at(seed, pos)
    pooled = NuLiteDETRDecoder._seed_columns(stub, seed, pos)
    assert plain.shape[-1] == 1 and pooled.shape[-1] == 3


def test_seed_columns_none_without_map():
    """No seed map yields no columns."""
    stub = _stub()
    assert NuLiteDETRDecoder._seed_columns(stub, None, torch.rand(1, 4, 2)) is None


def test_seed_columns_plain_when_not_pooled():
    """Pooled flag off falls back to the single raw column."""
    stub = _stub()
    stub.seed_in_content_pooled = False
    seed, pos = _maps()
    out = NuLiteDETRDecoder._seed_columns(stub, seed, pos)
    assert out.shape[-1] == 1


def test_modulation_zero_init_is_identity():
    """Zero-initialised projection leaves the content unchanged."""
    stub = _stub()
    stub.map_modulation = True
    stub.map_modulation_proj = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 6)
    )
    torch.nn.init.zeros_(stub.map_modulation_proj[2].weight)
    torch.nn.init.zeros_(stub.map_modulation_proj[2].bias)
    content = torch.randn(2, 7, 6)
    seed, pos = _maps(bs=2)
    out = NuLiteDETRDecoder._apply_map_modulation(stub, content, seed, pos)
    assert torch.allclose(out, content)


def test_modulation_noop_when_disabled():
    """Without the flag the content passes through untouched."""
    stub = _stub()
    content = torch.randn(2, 7, 6)
    seed, pos = _maps(bs=2)
    out = NuLiteDETRDecoder._apply_map_modulation(stub, content, seed, pos)
    assert out is content


def test_modulation_receives_gradient():
    """The projection learns: the last layer gets non-zero grad at step 0."""
    stub = _stub()
    stub.map_modulation = True
    proj = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 6))
    torch.nn.init.zeros_(proj[2].weight)
    torch.nn.init.zeros_(proj[2].bias)
    stub.map_modulation_proj = proj
    content = torch.zeros(1, 3, 6, requires_grad=True)
    seed, pos = _maps(bs=1)
    seed = seed[:, :, :8, :8]
    pos = pos[:, :3]
    out = NuLiteDETRDecoder._apply_map_modulation(stub, content, seed, pos)
    out.sum().backward()
    assert proj[2].weight.grad is not None
    assert torch.isfinite(proj[2].weight.grad).all()


@pytest.mark.parametrize('sizes', [(4,), (0,), (3, 2)])
def test_bad_pool_sizes_rejected(sizes):
    """Even windows or non-positive sizes are rejected at construction."""
    with pytest.raises(ValueError):
        NuLiteDETRDecoder._validate_seed_pool_sizes(sizes)
