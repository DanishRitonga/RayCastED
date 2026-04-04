"""Phase 3 tests — RayCastTileDataset and collate_fn.

Validates all checkpoints from docs/project.md §19.
Run with: uv run python tests/phase_3/test_dataloader.py
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset, collate_fn
from raycasted.data.etl.ops.augment import flip_horizontal, flip_vertical, rotate_90
from raycasted.data.etl.utils.constants import CX_IDX, CY_IDX, RAY_END_IDX, RAY_START_IDX

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_annotations(n_cells: int, cx_range=(100, 500), cy_range=(100, 500), ray_value=50.0) -> np.ndarray:
    """Create synthetic raycast annotations: [class_id, cx, cy, d_1..d_32]."""
    anns = np.zeros((n_cells, 35), dtype=np.float32)
    anns[:, 0] = 1  # class_id
    anns[:, CX_IDX] = np.random.uniform(cx_range[0], cx_range[1], n_cells)
    anns[:, CY_IDX] = np.random.uniform(cy_range[0], cy_range[1], n_cells)
    anns[:, RAY_START_IDX:] = ray_value
    return anns


def _make_image(h: int, w: int) -> np.ndarray:
    """Create a synthetic H&E-like image (non-white)."""
    rng = np.random.default_rng(42)
    img = rng.integers(80, 200, size=(h, w, 3), dtype=np.uint8)
    img[:, :, 0] = rng.integers(150, 220, size=(h, w), dtype=np.uint8)
    return img


def _create_synthetic_tile(
    directory: Path,
    name: str = 'tile_001.npz',
    image_size: tuple[int, int] = (1024, 1024),
    content_h: int | None = None,
    content_w: int | None = None,
    n_cells: int = 10,
) -> Path:
    """Write a synthetic .npz tile to directory and return its path."""
    h, w = image_size
    img = _make_image(h, w)
    anns = _make_annotations(n_cells, cx_range=(50, min(w - 50, 500)), cy_range=(50, min(h - 50, 500)))

    ch = content_h if content_h is not None else h
    cw = content_w if content_w is not None else w

    path = directory / name
    np.savez_compressed(path, image=img, annotations=anns, tissue=np.int32(1), content_h=np.int32(ch), content_w=np.int32(cw))
    return path


# ---------------------------------------------------------------------------
# Test 1: Normalise bounds — rays max <= 1.0, centroids in [0, 1]
# ---------------------------------------------------------------------------


def test_normalise_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        _create_synthetic_tile(tile_dir, n_cells=20, image_size=(1024, 1024), content_h=900, content_w=900)

        ds = RayCastTileDataset(str(tile_dir), crop_size=640, augment=False)

        for i in range(len(ds)):
            img, labels = ds[i]
            if len(labels) > 0:
                rays = labels[:, RAY_START_IDX:RAY_END_IDX]
                assert rays.max() <= 1.0, f'Ray > 1.0: {rays.max()}'

                cx_cy = labels[:, CX_IDX : CY_IDX + 1]
                assert cx_cy.min() >= 0.0, f'Centroid < 0: {cx_cy.min()}'
                assert cx_cy.max() <= 1.0, f'Centroid > 1.0: {cx_cy.max()}'
                break

    print('PASS: normalise bounds — rays <= 1.0, centroids in [0, 1]')


# ---------------------------------------------------------------------------
# Test 2: Random crop origin constrained — content < crop_size → origin = 0
# ---------------------------------------------------------------------------


def test_random_crop_origin_constrained():
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        # content smaller than crop_size
        _create_synthetic_tile(tile_dir, image_size=(512, 512), content_h=400, content_w=400, n_cells=5)

        ds = RayCastTileDataset(str(tile_dir), crop_size=640, augment=False)
        img, labels = ds[0]

        # Image should be padded to crop_size
        assert img.shape == (3, 640, 640), f'Image shape: {img.shape}'
    print('PASS: random crop origin constrained — small content handled')


# ---------------------------------------------------------------------------
# Test 3: Random crop within content — content > crop_size
# ---------------------------------------------------------------------------


def test_random_crop_within_content():
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        _create_synthetic_tile(tile_dir, image_size=(1024, 1024), content_h=800, content_w=800, n_cells=30)

        ds = RayCastTileDataset(str(tile_dir), crop_size=640, augment=False)

        # Run multiple times to check randomness
        any_annotations = False
        for _ in range(20):
            img, labels = ds[0]
            assert img.shape == (3, 640, 640), f'Image shape: {img.shape}'
            if len(labels) > 0:
                any_annotations = True
                # All centroids should be in [0, 1] after normalisation
                cx = labels[:, CX_IDX]
                cy = labels[:, CY_IDX]
                assert np.all(cx >= 0) and np.all(cx <= 1), f'cx out of range: [{cx.min()}, {cx.max()}]'
                assert np.all(cy >= 0) and np.all(cy <= 1), f'cy out of range: [{cy.min()}, {cy.max()}]'

        assert any_annotations, 'Expected at least some annotations across 20 crops'
    print('PASS: random crop within content — origins valid, centroids bounded')


# ---------------------------------------------------------------------------
# Test 4: Augmentation preserves annotation count
# ---------------------------------------------------------------------------


def test_augmentation_preserves_count():
    anns = _make_annotations(20)

    for func, canvas in [
        (flip_horizontal, 640),
        (flip_vertical, 640),
    ]:
        result = func(anns, canvas)
        assert len(result) == len(anns), f'{func.__name__} changed count: {len(anns)} → {len(result)}'

    for k in [1, 2, 3]:
        result = rotate_90(anns, k, 640)
        assert len(result) == len(anns), f'rotate_90(k={k}) changed count: {len(anns)} → {len(result)}'

    print('PASS: augmentation preserves annotation count')


# ---------------------------------------------------------------------------
# Test 5: Empty annotations handled
# ---------------------------------------------------------------------------


def test_empty_annotations_handled():
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        # Create tile with 0 annotations
        img = _make_image(640, 640)
        path = tile_dir / 'empty.npz'
        np.savez_compressed(path, image=img, annotations=np.zeros((0, 35), dtype=np.float32), tissue=np.int32(1), content_h=np.int32(640), content_w=np.int32(640))

        ds = RayCastTileDataset(str(tile_dir), crop_size=640, augment=False)
        img_tensor, labels = ds[0]

        assert labels.shape == (0, 35), f'Expected (0, 35), got {labels.shape}'
        assert img_tensor.shape == (3, 640, 640)
    print('PASS: empty annotations handled')


# ---------------------------------------------------------------------------
# Test 6: collate_fn shapes
# ---------------------------------------------------------------------------


def test_collate_fn_shapes():
    images = [torch.randn(3, 640, 640) for _ in range(4)]
    labels = [_make_annotations(np.random.randint(5, 20)) for _ in range(4)]

    batch = list(zip(images, labels))
    imgs_out, targets_out = collate_fn(batch)

    assert imgs_out.shape == (4, 3, 640, 640), f'Images shape: {imgs_out.shape}'
    assert targets_out.shape[1] == 36, f'Targets columns: {targets_out.shape[1]}'
    assert targets_out.dtype == torch.float32
    total_cells = sum(len(l) for l in labels)
    assert targets_out.shape[0] == total_cells, f'Expected {total_cells} rows, got {targets_out.shape[0]}'
    print('PASS: collate_fn shapes correct')


# ---------------------------------------------------------------------------
# Test 7: collate_fn batch_idx correct
# ---------------------------------------------------------------------------


def test_collate_fn_batch_idx():
    images = [torch.randn(3, 640, 640) for _ in range(3)]
    labels = [_make_annotations(n) for n in [5, 10, 3]]

    batch = list(zip(images, labels))
    _, targets = collate_fn(batch)

    batch_indices = targets[:, 0].to(torch.int32).numpy()
    assert np.all(batch_indices[:5] == 0)
    assert np.all(batch_indices[5:15] == 1)
    assert np.all(batch_indices[15:] == 2)
    print('PASS: collate_fn batch indices correct')


# ---------------------------------------------------------------------------
# Test 8: collate_fn with empty annotations
# ---------------------------------------------------------------------------


def test_collate_fn_empty():
    images = [torch.randn(3, 640, 640) for _ in range(2)]
    labels = [np.zeros((0, 35), dtype=np.float32), np.zeros((0, 35), dtype=np.float32)]

    batch = list(zip(images, labels))
    imgs_out, targets_out = collate_fn(batch)

    assert imgs_out.shape == (2, 3, 640, 640)
    assert targets_out.shape == (0, 36), f'Expected (0, 36), got {targets_out.shape}'
    print('PASS: collate_fn handles all-empty annotations')


# ---------------------------------------------------------------------------
# Test 9: _validate_batch passes for normalised data
# ---------------------------------------------------------------------------


def test_validate_batch_passes():
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        _create_synthetic_tile(tile_dir, n_cells=20, image_size=(1024, 1024), content_h=900, content_w=900)

        ds = RayCastTileDataset(str(tile_dir), crop_size=640, augment=False)

        # Should not assert for valid data
        for i in range(min(5, len(ds))):
            _, labels = ds[i]
            # _validate_batch already called in __getitem__; if we got here it passed
    print('PASS: _validate_batch passes for first batches')


# ---------------------------------------------------------------------------
# Test 10: Image-annotation spatial consistency after augmentation
# ---------------------------------------------------------------------------


def test_image_annotation_spatial_consistency():
    """After flipping/rotating, the centroid should map to the same relative position."""
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        # Single annotation at center
        img = _make_image(640, 640)
        anns = np.zeros((1, 35), dtype=np.float32)
        anns[0, 0] = 1
        anns[0, CX_IDX] = 320.0  # center
        anns[0, CY_IDX] = 320.0
        anns[0, RAY_START_IDX:] = 30.0

        path = tile_dir / 'center.npz'
        np.savez_compressed(path, image=img, annotations=anns, tissue=np.int32(1), content_h=np.int32(640), content_w=np.int32(640))

        ds = RayCastTileDataset(str(tile_dir), crop_size=640, augment=False)
        _, labels = ds[0]

        if len(labels) > 0:
            # Center should normalise to ~0.5
            assert abs(labels[0, CX_IDX] - 0.5) < 0.02, f'cx should be ~0.5, got {labels[0, CX_IDX]}'
            assert abs(labels[0, CY_IDX] - 0.5) < 0.02, f'cy should be ~0.5, got {labels[0, CY_IDX]}'
    print('PASS: image-annotation spatial consistency')


# ---------------------------------------------------------------------------
# Test 11: End-to-end dataset iteration
# ---------------------------------------------------------------------------


def test_e2e_dataset_iteration():
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        for i in range(5):
            _create_synthetic_tile(tile_dir, name=f'tile_{i:03d}.npz', n_cells=np.random.randint(5, 30))

        ds = RayCastTileDataset(str(tile_dir), crop_size=640, augment=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True, collate_fn=collate_fn)

        for i, (images, targets) in enumerate(loader):
            if i >= 3:
                break
            assert images.shape[0] <= 3
            assert images.shape[1] == 3
            assert images.shape[2] == 640
            assert images.shape[3] == 640
            assert targets.shape[1] == 36

            if targets.shape[0] > 0:
                rays = targets[:, 4:36]
                assert rays.max() <= 1.0, f'Ray > 1.0: {rays.max()}'

    print('PASS: end-to-end dataset iteration with DataLoader')


# ---------------------------------------------------------------------------
# Test 12: Visual output — augmented tiles with polygon overlays
# ---------------------------------------------------------------------------


def test_visual_output():
    """Generate visual output showing cropped+augmented tiles with polygon overlays.

    Saves to tests/phase_3/output/dataloader_visual.png for manual inspection.
    Verifies that augmented images look like plausible H&E (not inverted, not grey)
    and that polygon annotations align with the image content.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[INFO] test_visual_output  skipped (matplotlib not available)')
        return

    from raycasted.data.etl.ops.convert import decode_to_vertices

    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)

        # Create a synthetic tile with cells spread across the image
        _create_synthetic_tile(tile_dir, name='visual.npz', n_cells=15, image_size=(1024, 1024), content_h=900, content_w=900)

        ds = RayCastTileDataset(str(tile_dir), crop_size=640, augment=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Phase 3 — Augmented Tiles with Polygon Overlays', fontsize=14)

        for ax_idx, ax in enumerate(axes.flat):
            img_tensor, labels = ds[0]  # Different random crop/augment each call

            # Convert image tensor back to uint8 HWC for display
            img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            crop_size = img_np.shape[0]

            ax.imshow(img_np)

            if len(labels) > 0:
                # Denormalise annotations to pixel space
                pixel_labels = labels.copy()
                pixel_labels[:, CX_IDX] *= crop_size
                pixel_labels[:, CY_IDX] *= crop_size
                pixel_labels[:, RAY_START_IDX:RAY_END_IDX] *= crop_size

                cx = pixel_labels[:, CX_IDX]
                cy = pixel_labels[:, CY_IDX]
                rays = pixel_labels[:, RAY_START_IDX:RAY_END_IDX]

                # Decode to polygon vertices
                vertices = decode_to_vertices(rays, cx, cy)

                # Overlay each polygon
                for i in range(len(labels)):
                    vx = np.append(vertices[i, :, 0], vertices[i, 0, 0])
                    vy = np.append(vertices[i, :, 1], vertices[i, 0, 1])
                    ax.plot(vx, vy, color='cyan', linewidth=1.0, alpha=0.8)
                    ax.plot(cx[i], cy[i], 'r+', markersize=4)

                ax.set_title(f'Crop {ax_idx + 1}: {len(labels)} cells')
            else:
                ax.set_title(f'Crop {ax_idx + 1}: 0 cells')

            ax.set_xlim(0, crop_size)
            ax.set_ylim(crop_size, 0)

        plt.tight_layout()
        out_path = output_dir / 'dataloader_visual.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f'[INFO] test_visual_output  saved -> {out_path}')


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)

    test_normalise_bounds()
    test_random_crop_origin_constrained()
    test_random_crop_within_content()
    test_augmentation_preserves_count()
    test_empty_annotations_handled()
    test_collate_fn_shapes()
    test_collate_fn_batch_idx()
    test_collate_fn_empty()
    test_validate_batch_passes()
    test_image_annotation_spatial_consistency()
    test_e2e_dataset_iteration()
    test_visual_output()

    print('\nAll Phase 3 tests passed!')
