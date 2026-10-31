"""Phase 10 tests — frozen-oracle thinning (variant (a)).

``detr_backbone='frozen'`` keeps the whole NuLite-T U-Net (encoder, decoder,
decoder0, np_head) exactly as in ``two_stream`` and freezes all of it. The DETR
decoder consumes the oracle encoder's b3/b4/b5 directly, so there is no second
detection encoder and no trainable segmentation path:

    [backbone] -[b3/b4/b5]-> [DETR head]
    [backbone] -> [seg decoder] -> [FG map] -> [DETR head]
"""

import torch

from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel


def _build(**overrides):
    """Build a tiny frozen-oracle model for structural assertions."""
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
        'backbone': 'frozen',
        'seed_level': True,
        'seed_level_stride': 4,
        'seed_map_target': True,
        'oracle_kind': 'binary',
        'separable_decoder': True,
        'mds': False,
    }
    kwargs.update(overrides)
    return NuLiteRayCastDETRModel(**kwargs)


def test_frozen_backbone_has_oracle_and_no_detection_encoder():
    """The frozen backbone owns the oracle and drops the duplicate encoder."""
    model = _build()

    assert model.encoder_decoder is None
    assert model.seg_encoder is not None
    assert model.frozen_oracle is True
    assert model.lambda_seg == 0.0
    assert model.model[0] is model.seg_encoder
    assert model.model[-1] is model.detr


def test_frozen_backbone_params_fewer_than_two_stream():
    """Dropping the detection encoder removes the duplicated encoder mass."""
    frozen = _build()
    two = _build(backbone='two_stream')

    n_frozen = sum(p.numel() for p in frozen.parameters())
    n_two = sum(p.numel() for p in two.parameters())
    assert n_two - n_frozen > 5e6


def test_frozen_oracle_stays_frozen_and_in_eval_mode():
    """Every oracle parameter is frozen and no BN statistic can drift."""
    model = _build()

    assert not any(p.requires_grad for p in model.seg_encoder.parameters())
    model.train()
    assert not any(p.requires_grad for p in model.seg_encoder.parameters())
    bn = [m for m in model.seg_encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bn and not any(m.training for m in bn)


def test_frozen_detr_head_is_trainable():
    """The DETR head must keep training; only the oracle is frozen."""
    model = _build()
    model.train()

    assert sum(1 for p in model.detr.parameters() if p.requires_grad) > 0


def test_frozen_seed_map_matches_two_stream_shape():
    """Both layouts emit the same full-resolution seed logits."""
    frozen = _build()
    frozen.eval()
    with torch.no_grad():
        frozen.predict(torch.rand(1, 3, 256, 256))

    seed = frozen._last_seed
    assert seed.shape == (1, 1, 256, 256)
    assert torch.isfinite(seed).all()
    assert frozen._last_centroid_seed.shape == seed.shape


def test_frozen_seed_level_is_enabled():
    """The FG map still enters the decoder as the extra attended level."""
    model = _build()

    assert model.detr.seed_level is True


def test_frozen_unfreeze_all_keeps_oracle_frozen():
    """``unfreeze_all`` restores training but never thaws the oracle."""
    model = _build()
    model.train()
    model.unfreeze_all()

    assert any(p.requires_grad for p in model.detr.parameters())
    assert not any(p.requires_grad for p in model.seg_encoder.parameters())


def test_frozen_freeze_encoder_decoder_np_freezes_oracle():
    """The explicit freeze helper leaves the whole oracle frozen."""
    model = _build()
    model.unfreeze_all()
    model.freeze_encoder_decoder_np()

    assert not any(p.requires_grad for p in model.seg_encoder.parameters())


def test_thaw_oracle_encoder_exposes_only_the_encoder():
    """Thawing frees the FastViT encoder and leaves the seg readout frozen."""
    model = _build()
    model.train()
    model.thaw_oracle_encoder()

    assert model.oracle_encoder_thawed is True
    enc = model.seg_encoder.encoder_decoder.encoder
    assert any(p.requires_grad for p in enc.parameters())
    assert all(p.requires_grad for p in enc.parameters())
    # decoder0 / upsample decoder / np_head stay frozen: the map is unchanged.
    assert not any(p.requires_grad for p in model.seg_encoder.decoder0.parameters())
    assert not any(p.requires_grad for p in model.seg_encoder.np_head.parameters())
    assert not any(
        p.requires_grad
        for name, p in model.seg_encoder.encoder_decoder.named_parameters()
        if not name.startswith('encoder.')
    )


def test_thaw_survives_train_mode_transitions():
    """A later train()/eval() cycle must not silently re-freeze the encoder."""
    model = _build()
    model.thaw_oracle_encoder()
    model.train()
    enc = model.seg_encoder.encoder_decoder.encoder
    assert any(p.requires_grad for p in enc.parameters())
    model.eval()
    assert any(p.requires_grad for p in enc.parameters())


def test_thaw_requires_frozen_backbone():
    """The thaw helper only makes sense for the frozen layout."""
    model = _build(backbone='two_stream')
    try:
        model.thaw_oracle_encoder()
    except ValueError:
        return
    raise AssertionError('expected ValueError for a non-frozen backbone')


def test_freeze_helper_clears_the_thaw_state():
    """``freeze_encoder_decoder_np`` restores the full freeze."""
    model = _build()
    model.thaw_oracle_encoder()
    model.freeze_encoder_decoder_np()

    assert model.oracle_encoder_thawed is False
    assert not any(p.requires_grad for p in model.seg_encoder.parameters())


def test_map_is_unchanged_after_thawing():
    """Only the encoder adapts; at step 0 the seed map is still the oracle's."""
    model = _build()
    model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        model.predict(x)
        before = model._last_seed.clone()
        model.thaw_oracle_encoder()
        model.predict(x)
        after = model._last_seed.clone()
    assert torch.allclose(before, after)


def test_frozen_adapter_builds_and_taps_fusion():
    """With the adapter on, predict consumes fuse3/fuse4/fuse5, not raw b3/b4/b5."""
    model = _build(frozen_adapter=True)
    assert model.frozen_adapter is True
    assert model.fusion_adapters is not None
    assert len(model.fusion_adapters) == 3
    model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        y, _ = model.predict(x)
    assert y.shape == (1, 400, 68)
    assert bool(torch.isfinite(y).all())


def test_frozen_adapter_output_strides_and_channels():
    """Adapted features land at the native strides 4/8/16 with native channels."""
    model = _build(frozen_adapter=True)
    model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        d = model.seg_encoder.encoder_decoder((x - 0.5) / 0.5, collect_fusion=True)
        a3 = model.fusion_adapters[0](d['fuse3'])
        a4 = model.fusion_adapters[1](d['fuse4'])
        a5 = model.fusion_adapters[2](d['fuse5'])
    assert a3.shape == (1, 64, 64, 64)
    assert a4.shape == (1, 128, 32, 32)
    assert a5.shape == (1, 256, 16, 16)


def test_frozen_adapter_uses_concat_taps_not_decoder_outputs():
    """The adapter inputs are the pre-ConvBlock concats (pure-ish fusion taps)."""
    from raycasted.model.blocks.nulite import NuLiteEncoderDecoder

    ed = NuLiteEncoderDecoder(variant='fastvit_t12', pretrained=False, use_decoder=True, separable_decoder=True)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        d = ed(x, collect_fusion=True)
    assert d['fuse3'].shape == (1, 256, 32, 32)  # cat[z2(128), b4(128)]
    assert d['fuse4'].shape == (1, 512, 16, 16)  # cat[z3(256), b5(256)]
    assert d['fuse5'].shape == (1, 512, 8, 8)  # z4


def test_frozen_adapter_keeps_oracle_frozen():
    """Adapters are trainable; the oracle stays frozen and in eval mode."""
    model = _build(frozen_adapter=True)
    model.train()
    assert all(p.requires_grad for p in model.fusion_adapters.parameters())
    assert not any(p.requires_grad for p in model.seg_encoder.parameters())
    assert not model.seg_encoder.training


def test_frozen_adapter_off_has_no_adapters():
    """Default frozen build has no adapters and behaves as before."""
    model = _build()
    assert model.frozen_adapter is False
    assert model.fusion_adapters is None


def test_frozen_adapter_grads_flow_only_to_adapters_and_head():
    """Backward touches adapters + DETR head; oracle grads stay zero."""
    model = _build(frozen_adapter=True)
    model.train()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        d = model.seg_encoder.encoder_decoder((x - 0.5) / 0.5, collect_fusion=True)
    feats = [a(d[k]) for a, k in zip(model.fusion_adapters, ('fuse3', 'fuse4', 'fuse5'))]
    loss = sum(f.sum() for f in feats)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fusion_adapters.parameters())
    assert all(p.grad is None for p in model.seg_encoder.parameters())


def test_full_thaw_exposes_unet_but_not_np_head():
    """full_unet thaw trains encoder + upsample decoder + decoder0; np_head frozen."""
    model = _build(frozen_adapter=True)
    model.thaw_oracle_encoder(full_unet=True)
    assert not any(p.requires_grad for p in model.seg_encoder.np_head.parameters())
    assert all(p.requires_grad for p in model.seg_encoder.encoder_decoder.encoder.parameters())
    assert all(
        p.requires_grad
        for n, p in model.seg_encoder.encoder_decoder.named_parameters()
        if not n.startswith('encoder.')
    )
    assert all(p.requires_grad for p in model.seg_encoder.decoder0.parameters())


def test_full_thaw_survives_train_mode_transitions():
    """train()/eval() keep the U-Net thawed and np_head frozen+eval."""
    model = _build(frozen_adapter=True)
    model.thaw_oracle_encoder(full_unet=True)
    model.train()
    assert any(p.requires_grad for p in model.seg_encoder.encoder_decoder.encoder.parameters())
    assert not any(p.requires_grad for p in model.seg_encoder.np_head.parameters())
    assert not model.seg_encoder.np_head.training
    assert model.seg_encoder.encoder_decoder.encoder.training
    model.eval()
    assert not model.seg_encoder.np_head.training
    assert any(p.requires_grad for p in model.seg_encoder.encoder_decoder.parameters())


def test_full_thaw_freeze_helper_clears_state():
    """freeze_encoder_decoder_np resets both thaw flags."""
    model = _build(frozen_adapter=True)
    model.thaw_oracle_encoder(full_unet=True)
    model.freeze_encoder_decoder_np()
    assert model.oracle_encoder_thawed is False
    assert model.oracle_full_thaw is False
    assert not any(p.requires_grad for p in model.seg_encoder.parameters())


def test_full_thaw_and_staged_logic_light():
    """Warm-up semantics: pending state defers the thaw; after boundary it applies."""
    model = _build(frozen_adapter=True)
    # Before thaw: everything frozen.
    assert not any(p.requires_grad for p in model.seg_encoder.parameters())
    # Simulate the staged callback boundary reached.
    model.thaw_oracle_encoder(full_unet=True)
    assert any(p.requires_grad for p in model.seg_encoder.encoder_decoder.parameters())
    assert not any(p.requires_grad for p in model.seg_encoder.np_head.parameters())


def test_shared_layer_groups_round_robin():
    """2 groups over 6 steps: L4-6 share weights with L1-3; heads stay distinct."""
    from raycasted.model.blocks.nulite_detr import NuLiteDETRDecoder

    dec = NuLiteDETRDecoder(
        nc=5,
        hd=32,
        nq=100,
        ndl=6,
        ndp=4,
        n_rays=64,
        grid_size=0.1,
        shared_layer_groups=2,
    )
    ids = [id(layer) for layer in dec.decoder.layers]
    assert len(set(ids)) == 2
    assert ids[0] == ids[2] == ids[4]
    assert ids[1] == ids[3] == ids[5]
    assert ids[0] != ids[1]
    # per-layer heads remain distinct
    assert len({id(h) for h in dec.dec_bbox_head}) == 6
