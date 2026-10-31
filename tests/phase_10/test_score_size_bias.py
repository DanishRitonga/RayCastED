"""Focused tests for the learned per-query score-size bias."""

import torch
import torch.nn as nn

from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder, ScoreBias

NC = 5
N_OUT = NC + 1


def _bias():
    """Build a default ScoreBias module."""
    return ScoreBias(N_OUT, hidden=32)


def test_score_bias_zero_init_outputs_zero():
    """A fresh ScoreBias must be exactly zero for any input."""
    b = _bias()
    r = torch.tensor([[0.01], [0.1], [0.5], [1.0], [3.0]])
    out = b(r.unsqueeze(0))
    assert out.shape == (1, 5, N_OUT)
    assert out.abs().max().item() == 0.0


def test_score_bias_receives_gradient():
    """The last layer must collect gradient even though it starts at zero."""
    b = _bias()
    r = torch.rand(2, 7, 1) + 0.01
    out = b(r)
    out.sum().backward()
    assert b.net[-1].weight.grad is not None
    assert b.net[-1].weight.grad.abs().max().item() > 0


def test_score_bias_is_size_monotone_after_training_step():
    """A step pushing small sizes up must raise small-size logits more."""
    b = _bias()
    opt = torch.optim.SGD(b.parameters(), lr=0.002)
    small = torch.full((1, 4, 1), 0.02)
    large = torch.full((1, 4, 1), 2.0)
    for _ in range(10):
        opt.zero_grad()
        # want small queries to have HIGHER logits (recovering suppressed nuclei)
        loss = -(b(small).mean() - b(large).mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert b(small).mean() > b(large).mean()


def test_size_proxy_is_mean_ray_length():
    """_size_proxy returns the mean predicted ray length per query."""
    poly = torch.zeros(1, 2, 66)
    poly[0, 0, 2:] = 0.2
    poly[0, 1, 2:] = 0.8
    out = NuLiteDETRDecoder._size_proxy(poly)
    assert out.shape == (1, 2, 1)
    assert torch.allclose(out[0, 0, 0], torch.tensor(0.2))
    assert torch.allclose(out[0, 1, 0], torch.tensor(0.8))


def _decoder_stub(bias_enabled):
    """Minimal attribute stub for the score-bias helpers."""
    stub = nn.Module()
    stub._size_proxy = NuLiteDETRDecoder._size_proxy
    stub.score_size_bias = bias_enabled
    stub.enc_score_bias = ScoreBias(N_OUT, 32) if bias_enabled else None
    stub.dec_score_bias = nn.ModuleList([ScoreBias(N_OUT, 32) for _ in range(3)]) if bias_enabled else None
    return stub


def test_score_bias_enc_noop_when_disabled():
    """With the flag off, the encoder helper must return scores unchanged."""
    stub = _decoder_stub(False)
    scores = torch.randn(1, 400, N_OUT)
    poly = torch.rand(1, 400, 66) + 0.01
    out = NuLiteDETRDecoder._score_bias_enc(stub, scores, poly)
    assert torch.equal(out, scores)


def test_score_bias_enc_noop_at_init():
    """With the flag on but zero weights, the encoder adjustment is zero."""
    stub = _decoder_stub(True)
    scores = torch.randn(1, 400, N_OUT)
    poly = torch.rand(1, 400, 66) + 0.01
    out = NuLiteDETRDecoder._score_bias_enc(stub, scores, poly)
    assert torch.allclose(out, scores)


def test_score_bias_dec_uses_layer_specific_module():
    """Each decoder layer must use its own bias module."""
    stub = _decoder_stub(True)
    with torch.no_grad():
        stub.dec_score_bias[1].net[-1].weight.fill_(0.5)
    scores = torch.zeros(1, 4, N_OUT)
    poly = torch.full((1, 4, 66), 0.3)
    out0 = NuLiteDETRDecoder._score_bias_dec(stub, 0, scores, poly)
    out1 = NuLiteDETRDecoder._score_bias_dec(stub, 1, scores, poly)
    assert out0.abs().max().item() == 0.0
    assert out1.abs().max().item() > 0.0


def test_requires_no_object():
    """score_size_bias needs the explicit empty column to bias against."""
    import pytest

    from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel

    with pytest.raises(ValueError):
        NuLiteRayCastDETRModel(
            nc=NC,
            n_rays=64,
            backbone='nulite',
            pretrained=False,
            nq=40,
            ndl=1,
            hd=32,
            d_ffn=64,
            no_object=False,
            score_size_bias=True,
        )
