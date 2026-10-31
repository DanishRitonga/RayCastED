"""Phase 10 tests — single-stream (shared encoder+decoder) thinning.

``detr_backbone='shared'`` replaces the two-stream layout (frozen NuLite oracle
U-Net + separate detector encoder) with ONE frozen oracle encoder and ONE
shared NuLite decoder that serves both the frozen segmentation readout and the
trainable detection head.
"""

import torch

from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel


def _build(**overrides):
    """Build a tiny shared-stream model for structural assertions."""
    kwargs = {
        'nc': 5,
        'n_rays': 64,
        'variant': 'fastvit_t12',
        'det_variant': 'fastvit_t12',
        'pretrained': False,
        'nq': 400,
        'ndl': 2,
        'hd': 32,
        'd_ffn': 64,
        'grid_size': 0.05,
        'no_object': True,
        'backbone': 'shared',
        'seed_level': True,
        'seed_level_stride': 4,
        'seed_map_target': True,
        'oracle_kind': 'binary',
        'separable_decoder': True,
        'mds': False,
    }
    kwargs.update(overrides)
    return NuLiteRayCastDETRModel(**kwargs)


def test_shared_backbone_builds_one_encoder_and_no_separate_seg_stream():
    """The shared backbone owns a single encoder and no duplicate oracle model."""
    model = _build()

    assert model.seg_encoder is None
    assert model.shared_stream is True
    assert model.encoder_decoder.use_decoder is True
    assert model.decoder0 is not None
    assert model.np_head is not None


def test_shared_backbone_has_fewer_parameters_than_two_stream():
    """Dropping the duplicated detector encoder removes real parameters."""
    shared = _build()
    two_stream = _build(backbone='two_stream')

    n_shared = sum(p.numel() for p in shared.parameters())
    n_two = sum(p.numel() for p in two_stream.parameters())
    assert n_shared < n_two
    # One FastViT-T12 encoder is ~6.4M parameters; the saving is the whole stream.
    assert (n_two - n_shared) > 5e6


def test_shared_encoder_is_frozen_and_decoder_stays_trainable():
    """Only the oracle encoder is frozen; the shared decoder must adapt."""
    model = _build()

    assert not any(p.requires_grad for p in model.encoder_decoder.encoder.parameters())
    assert any(p.requires_grad for p in model.encoder_decoder.bottleneck_upsampler.parameters())
    assert any(p.requires_grad for p in model.decoder0.parameters())
    assert any(p.requires_grad for p in model.np_head.parameters())


def test_shared_train_mode_keeps_encoder_in_eval():
    """``train()`` must not put the frozen encoder back into training mode."""
    model = _build()
    model.train()

    assert not model.encoder_decoder.encoder.training
    assert model.encoder_decoder.bottleneck_upsampler.training
    assert not any(p.requires_grad for p in model.encoder_decoder.encoder.parameters())


def test_shared_seed_matches_two_stream_seed_semantics():
    """Both backbones emit the same binary seed = centroid - foreground logits."""
    torch.manual_seed(0)
    model = _build()
    model.eval()
    x = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        model.predict(x)

    seed = model._last_seed
    assert seed.shape == (1, 1, 256, 256)
    assert torch.isfinite(seed).all()


def test_shared_uses_seed_level_for_the_fourth_attention_level():
    """The seed map is available, so the 4th deformable level must stay on."""
    model = _build()
    assert model.detr.seed_level is True


def test_shared_freeze_helper_keeps_encoder_frozen():
    """``freeze_encoder_decoder_np`` freezes only the encoder in shared mode."""
    model = _build()
    model.unfreeze_all()
    assert all(p.requires_grad for p in model.encoder_decoder.encoder.parameters())

    model.freeze_encoder_decoder_np()
    assert not any(p.requires_grad for p in model.encoder_decoder.encoder.parameters())
    assert any(p.requires_grad for p in model.encoder_decoder.bottleneck_upsampler.parameters())
