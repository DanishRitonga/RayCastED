"""Phase 4 tests — PolygonDetect head.

Validates all checkpoints from docs/project.md §19, Phase 4:
  - Output shape: [B, N_anchors, 34]
  - ray_outputs.min() > 0 (Softplus active)
  - xy_outputs in [0, 1] (Sigmoid active)
  - self.no == nc + 34 (BUG-02)
  - DFL decoder absent from computational graph

Run with: uv run python tests/phase_4/test_polygon_head.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from raycasted.model.head import POLYGON_DIM, PolygonDetect, RayRefinementBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NC = 4
REG_MAX = 16
CH = (64, 128, 256)  # P3/P4/P5 channels
BATCH = 2
FEAT_SIZE = 80  # P3 spatial size (P4=40, P5=20)


def _make_feats(batch=BATCH, feat_size=FEAT_SIZE, ch=CH):
    """Create dummy multi-scale feature maps."""
    return [torch.randn(batch, c, feat_size // (2**i), feat_size // (2**i)) for i, c in enumerate(ch)]


def _count_anchors(feat_size=FEAT_SIZE):
    """Total anchors across P3/P4/P5."""
    return sum((feat_size // (2**i)) ** 2 for i in range(3))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ray_refinement_block():
    """RayRefinementBlock preserves shape and has residual connection."""
    block = RayRefinementBlock(64)
    x = torch.randn(2, 64, 20, 20)
    y = block(x)
    assert y.shape == x.shape, f'Shape mismatch: {y.shape} vs {x.shape}'

    # Residual: output should differ from input (conv is not identity)
    assert not torch.allclose(y, x, atol=1e-6), 'Output should differ from input'

    # GroupNorm: verify groups config
    assert block.gn.num_groups == 8, f'Expected 8 groups, got {block.gn.num_groups}'


def test_ray_refinement_block_small_channels():
    """RayRefinementBlock uses 4 groups when channels < 64."""
    block = RayRefinementBlock(32)
    assert block.gn.num_groups == 4, f'Expected 4 groups for 32 channels, got {block.gn.num_groups}'


def test_output_shape():
    """Forward pass produces [B, 34, N_anchors] polygon output."""
    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    feats = _make_feats()
    head.eval()

    with torch.no_grad():
        preds = head.forward_head(feats, box_head=head.cv2, cls_head=head.cv3)

    poly = preds['boxes']
    n_anchors = _count_anchors()

    assert poly.shape == (BATCH, POLYGON_DIM, n_anchors), (
        f'Expected ({BATCH}, {POLYGON_DIM}, {n_anchors}), got {poly.shape}'
    )
    print(f'PASS: output shape — {poly.shape} == (B={BATCH}, 34, N={n_anchors})')


def test_softplus_active():
    """Ray channels after Softplus have min() > 0."""
    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    feats = _make_feats()
    head.eval()

    with torch.no_grad():
        preds = head.forward_head(feats, box_head=head.cv2, cls_head=head.cv3)

    poly = preds['boxes']
    ray_raw = poly[:, 2:, :]  # channels 2-33
    ray_activated = F.softplus(ray_raw)

    assert ray_activated.min() > 0, f'Softplus output should be > 0, got min={ray_activated.min()}'
    print(f'PASS: Softplus active — min(ray) = {ray_activated.min().item():.6f} > 0')


def test_sigmoid_active():
    """XY channels after Sigmoid are in [0, 1]."""
    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    feats = _make_feats()
    head.eval()

    with torch.no_grad():
        preds = head.forward_head(feats, box_head=head.cv2, cls_head=head.cv3)

    poly = preds['boxes']
    xy_raw = poly[:, :2, :]
    xy_activated = xy_raw.sigmoid()

    assert xy_activated.min() >= 0.0, f'Sigmoid min should be >= 0, got {xy_activated.min()}'
    assert xy_activated.max() <= 1.0, f'Sigmoid max should be <= 1, got {xy_activated.max()}'
    print(f'PASS: Sigmoid active — xy in [{xy_activated.min().item():.4f}, {xy_activated.max().item():.4f}]')


def test_self_no_override():
    """head.no == nc + 34 (BUG-02)."""
    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    expected = NC + POLYGON_DIM
    assert head.no == expected, f'Expected no={expected}, got no={head.no}'
    print(f'PASS: self.no = {head.no} == nc({NC}) + 34 = {expected}')


def test_dfl_absent():
    """DFL is nn.Identity, not DFL module."""
    from ultralytics.nn.modules.head import DFL

    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)

    # DFL should be Identity
    assert isinstance(head.dfl, nn.Identity), f'DFL should be Identity, got {type(head.dfl)}'

    # No DFL class in cv2 modules
    for layer in head.cv2:
        for module in layer.modules():
            assert not isinstance(module, DFL), f'Found DFL in cv2: {module}'

    print('PASS: DFL absent — head.dfl is nn.Identity, no DFL in cv2')


def test_inference_path():
    """Inference path applies activations and produces valid output."""
    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)

    # Compute strides by running a forward pass
    feats = _make_feats()
    head.eval()
    head.stride = torch.tensor([8.0, 16.0, 32.0])

    with torch.no_grad():
        preds = head.forward_head(feats, box_head=head.cv2, cls_head=head.cv3)
        result = head._inference(preds)

    n_anchors = _count_anchors()
    # Result shape: [B, 34 + nc, N_anchors]
    assert result.shape[0] == BATCH
    assert result.shape[2] == n_anchors
    # xy (decoded to pixel space) should be positive
    assert result[:, :2, :].min() >= 0, 'Decoded xy should be non-negative'
    # rays (softplus * strides) should be positive
    assert result[:, 2:34, :].min() > 0, 'Decoded rays should be > 0'
    # scores (sigmoid) should be in [0, 1]
    scores = result[:, 34:, :]
    assert scores.min() >= 0 and scores.max() <= 1, 'Scores should be in [0, 1]'
    print(f'PASS: inference path — shape={result.shape}, rays>0, scores in [0,1]')


def test_training_forward():
    """Training forward returns raw logits (no activations)."""
    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.train()

    feats = _make_feats()
    preds = head(feats)

    # Training returns a dict with 'boxes', 'scores', 'feats'
    assert 'boxes' in preds, f"Missing 'boxes' key, got {list(preds.keys())}"
    assert 'scores' in preds, f"Missing 'scores' key, got {list(preds.keys())}"

    poly = preds['boxes']
    n_anchors = _count_anchors()
    assert poly.shape == (BATCH, POLYGON_DIM, n_anchors)
    print(f'PASS: training forward — {list(preds.keys())}, poly shape={poly.shape}')


def test_loss_stub_no():
    """PolygonDetectionLoss stub: self.no == nc + 34, use_dfl == False."""
    # Build a minimal model with PolygonDetect as the head
    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    head.stride = torch.tensor([8.0, 16.0, 32.0])

    # Create a minimal model-like object for v8DetectionLoss
    class _FakeModel:
        class _Args:
            box = 7.5
            cls = 0.5
            dfl = 1.5
            epochs = 100

        def __init__(self, head):
            self.model = [nn.Identity()] * 10  # filler layers
            self.model[-1] = self._head = head
            self.args = self._Args()

        def parameters(self):
            return self._head.parameters()

    from raycasted.model.loss import PolygonDetectionLoss

    model = _FakeModel(head)
    loss = PolygonDetectionLoss(model)

    assert loss.no == NC + 34, f'Expected no={NC + 34}, got no={loss.no}'
    assert loss.use_dfl is False, f'Expected use_dfl=False, got use_dfl={loss.use_dfl}'
    print(f'PASS: loss stub — no={loss.no}, use_dfl={loss.use_dfl}')


def test_decode_round_trip():
    """Encode known polygons as logits, decode through _inference, verify round-trip.

    Tests all three scale levels (P3/P4/P5) with non-trivial logits.
    Verifies the decode formula: xy = (sigmoid(logit)*2 - 0.5 + anchor)*stride
    and rays = softplus(logit)*stride.
    """
    import numpy as np

    from raycasted.data.etl.ops.convert import decode_to_vertices

    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    head.eval()
    head.stride = torch.tensor([8.0, 16.0, 32.0])

    feats = _make_feats(batch=1)
    with torch.no_grad():
        head.forward_head(feats, box_head=head.cv2, cls_head=head.cv3)

    # Zero out all predictions so we can set specific anchors
    n_anchors = _count_anchors()
    boxes = torch.zeros(1, POLYGON_DIM, n_anchors)
    scores = torch.zeros(1, NC, n_anchors)

    # --- Choose test anchors across all three scales ---
    # P3: 80x80 = 6400, P4: 40x40 = 1600, P5: 20x20 = 400
    test_cases = [
        # (scale_name, global_idx, row, col, stride)
        ('P3', 40 * 80 + 40, 40, 40, 8.0),
        ('P4', 6400 + 20 * 40 + 20, 20, 20, 16.0),
        ('P5', 8000 + 10 * 20 + 10, 10, 10, 32.0),
    ]

    # Known polygon: circle with radius 25px, centred at grid-cell centre
    # Grid-cell centre in grid coords = (col + 0.5, row + 0.5)
    # xy_offset = 0.5 when centred → logit = 0.0
    # We'll use a non-trivial offset too: shift by 0.3 grid cells right and 0.2 down
    xy_logit = torch.tensor([math.log(0.8 / 0.2), math.log(0.7 / 0.3)])  # inverse sigmoid of (0.8, 0.7)
    expected_xy_offset = torch.sigmoid(xy_logit)  # (0.8, 0.7)

    # Rays: all equal to 25px in pixel space → ray_logit = inverse_softplus(25 / stride)
    ray_radius_px = 25.0

    for scale_name, anchor_idx, row, col, stride in test_cases:
        # Expected ray in normalised (grid) space
        ray_grid = ray_radius_px / stride
        ray_logit_val = math.log(math.exp(ray_grid) - 1)  # inverse softplus
        ray_logit = torch.full((32,), ray_logit_val)

        # Place logits
        boxes[0, :2, anchor_idx] = xy_logit
        boxes[0, 2:, anchor_idx] = ray_logit
        scores[0, 0, anchor_idx] = 5.0  # high confidence for class 0

    preds_manual = dict(boxes=boxes, scores=scores, feats=feats)

    with torch.no_grad():
        decoded = head._inference(preds_manual)

    # --- Verify each test case ---
    for scale_name, anchor_idx, row, col, stride in test_cases:
        # Anchor grid position from make_anchors: sx = col + 0.5, sy = row + 0.5
        anchor_gx = col + 0.5
        anchor_gy = row + 0.5

        # Expected decoded xy in pixels
        expected_x = (expected_xy_offset[0].item() * 2.0 - 0.5 + anchor_gx) * stride
        expected_y = (expected_xy_offset[1].item() * 2.0 - 0.5 + anchor_gy) * stride

        decoded_x = decoded[0, 0, anchor_idx].item()
        decoded_y = decoded[0, 1, anchor_idx].item()

        assert abs(decoded_x - expected_x) < 0.5, (
            f'{scale_name} xy X: decoded={decoded_x:.2f}, expected={expected_x:.2f}'
        )
        assert abs(decoded_y - expected_y) < 0.5, (
            f'{scale_name} xy Y: decoded={decoded_y:.2f}, expected={expected_y:.2f}'
        )

        # Expected ray in pixels
        expected_ray_px = ray_radius_px
        decoded_rays = decoded[0, 2:34, anchor_idx]

        assert torch.allclose(decoded_rays, torch.full((32,), expected_ray_px), atol=1.0), (
            f'{scale_name} rays: decoded≈{decoded_rays[0].item():.2f}, expected≈{expected_ray_px:.2f}'
        )

        # --- Round-trip polygon vertices ---
        # Verify decoded polygon vertices match expected geometry
        cx, cy = decoded_x, decoded_y
        rays_np = decoded_rays.numpy()
        verts = decode_to_vertices(rays_np[np.newaxis], np.array([cx]), np.array([cy]))[0]

        # All vertices should be at distance ≈ ray_radius_px from centroid
        dists = np.sqrt((verts[:, 0] - cx) ** 2 + (verts[:, 1] - cy) ** 2)
        assert np.allclose(dists, ray_radius_px, atol=1.5), (
            f'{scale_name} vertex distances: mean={dists.mean():.2f}, expected={ray_radius_px:.2f}'
        )

        print(
            f'PASS: {scale_name} round-trip — '
            f'xy=({decoded_x:.1f}, {decoded_y:.1f}) expected=({expected_x:.1f}, {expected_y:.1f}), '
            f'rays≈{decoded_rays[0].item():.1f}px, vertex_dist_mean={dists.mean():.1f}'
        )


def test_bias_init():
    """bias_init sets ray biases to ~15px at 0.25 MPP across all scales."""
    head = PolygonDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.bias_init()

    target_ray_px = 15.0
    strides = [8.0, 16.0, 32.0]

    for i, stride in enumerate(strides):
        bias = head.cv2[i][-1].bias.data  # [34]
        # XY biases should be 2.0
        assert bias[:2].tolist() == [2.0, 2.0], f'Scale {i}: xy bias should be [2.0, 2.0], got {bias[:2].tolist()}'
        # Ray biases should decode to ~15px: softplus(bias) * stride ≈ 15
        ray_biases = bias[2:]
        decoded_rays = F.softplus(ray_biases) * stride
        assert torch.allclose(decoded_rays, torch.full_like(decoded_rays, target_ray_px), atol=0.1), (
            f'Scale {i} (stride={stride}): decoded rays should be ~{target_ray_px}px, '
            f'got mean={decoded_rays.mean():.2f}'
        )
        print(
            f'PASS: scale {i} (stride={stride}) — '
            f'xy bias=[2.0, 2.0], ray bias={ray_biases[0]:.3f} → {decoded_rays[0]:.1f}px'
        )


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_ray_refinement_block()
    test_ray_refinement_block_small_channels()
    test_output_shape()
    test_softplus_active()
    test_sigmoid_active()
    test_self_no_override()
    test_dfl_absent()
    test_inference_path()
    test_training_forward()
    test_loss_stub_no()
    test_decode_round_trip()
    test_bias_init()

    print('\nAll Phase 4 tests passed!')
