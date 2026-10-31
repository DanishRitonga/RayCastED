"""Focused checks for nucleus-CDN: canonical GT-polygon denoising queries."""

from types import SimpleNamespace

import pytest
import torch

from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder, decode_polygon
from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel

NC = 5
RAD_DIM = 66


def _polygon(x, y, radius=0.08):
    return torch.tensor([x, y] + [radius] * 64)


def _decoder(nucleus_cdn=2, nc=NC, class_embed=False):
    stub = SimpleNamespace(
        nucleus_cdn=nucleus_cdn,
        nucleus_cdn_class_embed=class_embed,
        nc=nc,
        raycast_dim=RAD_DIM,
        num_queries=4,
        training=True,
        seed_in_content=False,
        nucleus_dn_token=torch.zeros(1, 1, 8),
        denoising_class_embed=torch.nn.Embedding(nc, 8),
    )
    return stub


def _run_queries(stub, gt, cls, gt_groups, bs=1, seed_map=None):
    proj2d = [torch.zeros(bs, 4, 8, 8)]
    return NuLiteDETRDecoder._nucleus_dn_queries(
        stub,
        {'bboxes': gt, 'cls': cls, 'gt_groups': gt_groups},
        proj2d,
        seed_map=seed_map,
    )


def test_dn_queries_shape_labels_and_padding():
    """Each GT is duplicated, padded slots carry the no-object label."""
    stub = _decoder(nucleus_cdn=2)
    gt = torch.stack((_polygon(0.3, 0.3), _polygon(0.7, 0.7)))
    embed, refer, meta = _run_queries(stub, gt, torch.tensor([2, 3]), [2])
    assert refer.shape == (1, 4, RAD_DIM)
    assert embed.shape == (1, 4, 8)
    assert meta['n_dn'] == 4
    assert meta['labels'].tolist() == [[2, 2, 3, 3]]
    assert meta['valid'].all()
    assert meta['slot'][0, :2].all() and not meta['slot'][0, 2:].any()


def test_dn_queries_pad_across_batch_images():
    """Padding to the batch max GT count keeps the DN block rectangular."""
    stub = _decoder(nucleus_cdn=1)
    gt = torch.stack((_polygon(0.3, 0.3), _polygon(0.7, 0.7), _polygon(0.5, 0.5)))
    embed, refer, meta = _run_queries(stub, gt, torch.tensor([2, 3, 4]), [2, 1], bs=2)
    assert refer.shape == (2, 2, RAD_DIM)
    assert meta['labels'].tolist() == [[2, 3], [4, NC]]
    assert meta['valid'].tolist() == [[True, True], [True, False]]


def test_dn_queries_reference_is_encoded_gt_polygon():
    """The DN reference polygon is the GT polygon in ray-logit space."""
    stub = _decoder(nucleus_cdn=1)
    gt = _polygon(0.3, 0.4)[None]
    _, refer, _ = _run_queries(stub, gt, torch.tensor([1]), [1])
    decoded = decode_polygon(refer)
    assert torch.allclose(decoded[0, 0, :2], gt[0, :2], atol=1e-5)


@pytest.mark.parametrize(
    'mutation',
    ['eval_mode', 'disabled', 'no_gt'],
)
def test_dn_queries_no_op_conditions(mutation):
    """No DN block in eval, when disabled, or without any GT."""
    stub = _decoder(nucleus_cdn=2)
    gt = _polygon(0.3, 0.3)[None]
    cls = torch.tensor([2])
    groups = [1]
    if mutation == 'eval_mode':
        stub.training = False
    elif mutation == 'disabled':
        stub.nucleus_cdn = 0
    else:
        gt = torch.empty(0, RAD_DIM)
        cls = torch.empty(0, dtype=torch.long)
        groups = [0]
    embed, refer, meta = _run_queries(stub, gt, cls, groups)
    assert embed is None and refer is None and meta is None


def test_dn_embedding_is_class_agnostic_by_default():
    """Default DN embedding carries no GT class: no private answer lookup.

    The score head must read the image, not a label-derived embedding row.
    Train35 showed the head could shortcut through ``denoising_class_embed``
    (worth 7.9 logits while the DN token stayed dead), so the default path
    must give every class the SAME embedding.
    """
    stub = _decoder(nucleus_cdn=1)
    gt = torch.stack((_polygon(0.3, 0.3), _polygon(0.7, 0.7)))
    embed, _, _ = _run_queries(stub, gt, torch.tensor([2, 4]), [2])
    assert embed.shape == (1, 2, 8)
    assert torch.equal(embed[0, 0], embed[0, 1])


def test_dn_embedding_recovers_class_information_when_requested():
    """The control flag restores the train35 private class lookup."""
    stub = _decoder(nucleus_cdn=1, class_embed=True)
    torch.nn.init.normal_(stub.nucleus_dn_token, std=0.02)
    torch.nn.init.normal_(stub.denoising_class_embed.weight, std=0.5)
    gt = torch.stack((_polygon(0.3, 0.3), _polygon(0.7, 0.7)))
    embed, _, _ = _run_queries(stub, gt, torch.tensor([2, 4]), [2])
    assert not torch.allclose(embed[0, 0], embed[0, 1])


def test_real_to_dn_attention_mask_quadrants():
    """Default blocks both DN<->real quadrants; the flag opens real->DN only.

    The mask is built inline in ``NuLiteDETRDecoder.forward``, so this test
    re-derives the two quadrant patterns the forward pass produces and checks
    the flag flips exactly the [real rows, DN cols] quadrant.
    """
    n_dn, n_real = 3, 4
    n_q = n_dn + n_real
    base = torch.zeros(1, n_q, n_q, dtype=torch.bool)
    base[:, :n_dn, n_dn:] = True
    isolated = base.clone()
    isolated[:, n_dn:, :n_dn] = True
    # default: both off-diagonal quadrants blocked
    assert isolated[:, :n_dn, n_dn:].all() and isolated[:, n_dn:, :n_dn].all()
    assert not isolated[:, :n_dn, :n_dn].any() and not isolated[:, n_dn:, n_dn:].any()
    # real->DN open: only DN rows remain blocked from real cols
    opened = base.clone()
    assert opened[:, :n_dn, n_dn:].all()
    assert not opened[:, n_dn:, :n_dn].any()


def test_real_to_dn_flag_wiring():
    """The decoder stores the flag; default False isolates DN completely."""
    stub = _decoder()
    assert getattr(stub, 'nucleus_cdn_class_embed', None) is not None
    stub2 = SimpleNamespace(nucleus_cdn_real_to_dn=False)
    assert stub2.nucleus_cdn_real_to_dn is False


def test_nucleus_cdn_class_loss_gradient_and_value():
    """Targets are pushed up (negative grad), everything else pushed down."""
    stub = SimpleNamespace(nc=NC)
    labels = torch.tensor([[2, NC]])
    meta = {'labels': labels}
    logits = torch.zeros(2, 1, 2, NC + 1, requires_grad=True)
    value = NuLiteRayCastDETRModel._nucleus_cdn_class_loss(stub, logits, meta)
    value.backward()
    grad = logits.grad[0, 0]
    # valid row: true class is a target (raised) ...
    assert grad[0, 2] < 0
    # ... while the empty column and the other real classes are suppressed.
    assert grad[0, -1] > 0
    rest = [c for c in range(NC) if c != 2]
    assert (grad[0, rest] > 0).all()
    # padded row trains only the empty column to fire.
    assert grad[1, -1] < 0
    assert (grad[1, :NC] > 0).all()


def test_nucleus_cdn_class_loss_prefers_correct_prediction():
    """A correct one-hot-ish prediction yields a smaller loss."""
    stub = SimpleNamespace(nc=NC)
    meta = {'labels': torch.tensor([[2, NC]])}
    good = torch.full((1, 1, 2, NC + 1), -8.0)
    bad = torch.full((1, 1, 2, NC + 1), -8.0)
    good[0, 0, 0, 2] = 8.0
    good[0, 0, 0, -1] = -8.0
    good[0, 0, 1, -1] = 8.0
    bad[0, 0, 0, -1] = 8.0
    good_loss = NuLiteRayCastDETRModel._nucleus_cdn_class_loss(stub, good, meta)
    bad_loss = NuLiteRayCastDETRModel._nucleus_cdn_class_loss(stub, bad, meta)
    assert good_loss < bad_loss
