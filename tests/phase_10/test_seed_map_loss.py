"""Consumer-4 ablation: the direct seed-map loss in frozen-prior layouts.

The ``frozen`` / ``two_stream`` layouts treat the segmentation stream as a
frozen prior, so ``lambda_seg`` is hard-zeroed and the seed map is only ever
trained through detector gradients after the epoch-50 thaw. ``seed_map_loss``
(recipe key ``detr_seed_map_loss``) opts back in to the direct
``L1(seed_map, batch['seed_map'])`` term that every recipe already configures
via ``lambda_seg``.

These tests pin:
  * default off keeps ``lambda_seg == 0.0`` (bit-identical old behaviour),
  * opt in honours the configured ``lambda_seg``,
  * the L1 term is really added to the total when enabled,
  * the oracle only receives gradients for it after the staged thaw.
"""

import torch
import torch.nn.functional as F

from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel

N_RAYS = 64
NQ = 16
IMG = 64


def _model(backbone='frozen', seed_map_loss=False, lambda_seg=1.0, **over):
    kwargs = dict(
        nc=5,
        n_rays=N_RAYS,
        variant='fastvit_t8',
        det_variant='fastvit_t8',
        backbone=backbone,
        pretrained=False,
        lambda_seg=lambda_seg,
        seed_map_loss=seed_map_loss,
        nq=NQ,
        ndl=2,
        hd=32,
        d_ffn=64,
        grid_size=0.25,
        verbose=False,
    )
    kwargs.update(over)
    return NuLiteRayCastDETRModel(**kwargs)


def _batch(bs=1, anns=3):
    torch.manual_seed(0)
    bboxes = torch.rand(anns, 2 + N_RAYS)
    bboxes[:, 0] = 0.2 + 0.6 * torch.rand(anns)
    bboxes[:, 1] = 0.2 + 0.6 * torch.rand(anns)
    bboxes[:, 2:] = 0.02 + 0.04 * torch.rand(anns, N_RAYS)
    seed_map = torch.zeros(bs, 1, IMG, IMG)
    seed_map[:, :, 10:20, 10:20] = 1.0
    np_mask = seed_map.clone()
    semantic_mask = torch.zeros(bs, IMG, IMG, dtype=torch.long)
    semantic_mask[:, 10:20, 10:20] = 1
    instance_mask = torch.zeros(bs, IMG, IMG, dtype=torch.long)
    instance_mask[:, 10:20, 10:20] = 1
    return {
        'img': torch.rand(bs, 3, IMG, IMG),
        'batch_idx': torch.zeros(anns, dtype=torch.long),
        'cls': torch.randint(0, 5, (anns,)),
        'bboxes': bboxes,
        'seed_map': seed_map,
        'np_mask': np_mask,
        'instance_mask': instance_mask,
        'semantic_mask': semantic_mask,
    }


def test_default_off_keeps_seg_loss_inert_in_frozen_layout():
    """Without the opt-in key the frozen prior must stay unsupervised."""
    model = _model(backbone='frozen')
    assert model.seed_map_loss is False
    assert model.lambda_seg == 0.0


def test_default_off_keeps_seg_loss_inert_in_two_stream_layout():
    """two_stream shares the frozen-prior contract and must match."""
    model = _model(backbone='two_stream')
    assert model.seed_map_loss is False
    assert model.lambda_seg == 0.0


def test_opt_in_honours_configured_lambda_seg():
    """Opting in reuses the recipe's existing lambda_seg weight."""
    assert _model(seed_map_loss=True, lambda_seg=1.0).lambda_seg == 1.0
    assert _model(seed_map_loss=True, lambda_seg=2.5).lambda_seg == 2.5


def test_seg_term_is_added_to_the_total_when_enabled():
    """The only difference between the two totals is lambda_seg * L1(seed, target)."""
    model = _model(seed_map_loss=True, lambda_seg=1.0)
    model.train()
    batch = _batch()

    model.lambda_seg = 0.0
    torch.manual_seed(1)
    off, _ = model.loss(batch)

    model.lambda_seg = 1.0
    torch.manual_seed(1)
    on, _ = model.loss(batch)

    with torch.no_grad():
        expected = F.l1_loss(model._last_seed, batch['seed_map'])
    assert expected.item() > 0.0
    assert torch.allclose(on - off, expected, atol=1e-4, rtol=1e-4), (
        on.item(),
        off.item(),
        expected.item(),
    )


def test_seg_term_only_reaches_the_oracle_after_the_thaw():
    """Pre-thaw the frozen U-Net gets no gradient; post-thaw the anchor is live."""
    model = _model(seed_map_loss=True, lambda_seg=1.0)
    model.train()
    batch = _batch()

    model.zero_grad(set_to_none=True)
    total, _ = model.loss(batch)
    total.backward()
    assert not any(p.grad is not None for p in model.seg_encoder.parameters())

    model.thaw_oracle_encoder(full_unet=True)
    model.zero_grad(set_to_none=True)
    total, _ = model.loss(batch)
    total.backward()
    oracle_grads = sum(1 for p in model.seg_encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert oracle_grads > 0
    assert not any(p.requires_grad for p in model.seg_encoder.np_head.parameters())
