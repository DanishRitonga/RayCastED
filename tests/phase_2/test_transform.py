"""Phase 2 tests — SpatialChunker raycast branch, NormalizerAndPadder, TransformOrchestrator.

Validates all checkpoints from docs/project.md §19.
"""

import tempfile
from pathlib import Path

import numpy as np

from raycasted.data.etl.ops.filter import filter_and_clip_annotations
from raycasted.data.etl.transform.normalizer import NormalizerAndPadder
from raycasted.data.etl.transform.spatialChunker import SpatialChunker
from raycasted.data.etl.transform.transform_orchestrator import TransformOrchestrator
from raycasted.data.etl.utils.constants import CX_IDX, CY_IDX, RAY_START_IDX

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_annotations(n_cells: int, cx_range=(100, 900), cy_range=(100, 900), ray_value=50.0) -> np.ndarray:
    """Create synthetic raycast annotations: [class_id, cx, cy, d_1..d_32]."""
    anns = np.zeros((n_cells, 35), dtype=np.float32)
    anns[:, 0] = 1  # class_id
    anns[:, CX_IDX] = np.random.uniform(cx_range[0], cx_range[1], n_cells)
    anns[:, CY_IDX] = np.random.uniform(cy_range[0], cy_range[1], n_cells)
    anns[:, RAY_START_IDX:] = ray_value
    return anns


def _make_image(h: int, w: int) -> np.ndarray:
    """Create a synthetic H&E-like image (non-white so Macenko works)."""
    rng = np.random.default_rng(42)
    # Pink-ish tissue colors to pass Macenko OD threshold
    img = rng.integers(80, 200, size=(h, w, 3), dtype=np.uint8)
    img[:, :, 0] = rng.integers(150, 220, size=(h, w), dtype=np.uint8)  # more red (eosin)
    img[:, :, 2] = rng.integers(120, 180, size=(h, w), dtype=np.uint8)  # some blue
    return img


# ---------------------------------------------------------------------------
# Test 1: NormalizerAndPadder.process_roi() returns 4 values
# ---------------------------------------------------------------------------


def test_normalizer_returns_4_values():
    config = {'max_size': 256}
    norm = NormalizerAndPadder(config, profile_path=None)

    img = _make_image(200, 200)
    anns = _make_annotations(5)
    result = norm.process_roi(img, anns)

    assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
    assert len(result) == 4, f'Expected 4 values, got {len(result)}'

    img_out, anns_out, content_h, content_w = result
    assert isinstance(content_h, int), f'content_h should be int, got {type(content_h)}'
    assert isinstance(content_w, int), f'content_w should be int, got {type(content_w)}'
    print('PASS: NormalizerAndPadder.process_roi() returns 4 values')


# ---------------------------------------------------------------------------
# Test 2: content_h/content_w are original dimensions before padding
# ---------------------------------------------------------------------------


def test_content_dims_before_padding():
    config = {'max_size': 512}
    norm = NormalizerAndPadder(config, profile_path=None)

    # Image smaller than target → will be padded
    img = _make_image(300, 400)
    anns = _make_annotations(3)
    img_out, anns_out, content_h, content_w = norm.process_roi(img, anns)

    assert content_h == 300, f'content_h should be 300, got {content_h}'
    assert content_w == 400, f'content_w should be 400, got {content_w}'
    assert img_out.shape[0] == 512, f'Padded height should be 512, got {img_out.shape[0]}'
    assert img_out.shape[1] == 512, f'Padded width should be 512, got {img_out.shape[1]}'
    print('PASS: content_h/content_w are original dimensions before padding')


# ---------------------------------------------------------------------------
# Test 3: Unpadded tiles have content_h == image.shape[0], content_w == image.shape[1]
# ---------------------------------------------------------------------------


def test_unpadded_content_dims():
    config = {'max_size': 256}
    norm = NormalizerAndPadder(config, profile_path=None)

    # Image already at target size → no padding
    img = _make_image(256, 256)
    anns = _make_annotations(2)
    img_out, anns_out, content_h, content_w = norm.process_roi(img, anns)

    assert content_h == img.shape[0] == 256
    assert content_w == img.shape[1] == 256
    print('PASS: Unpadded tiles have content_h == image.shape[0], content_w == image.shape[1]')


# ---------------------------------------------------------------------------
# Test 4: SpatialChunker raycast branch — centroid stays inside chunk
# ---------------------------------------------------------------------------


def test_spatial_chunker_raycast_centroid_in_bounds():
    config = {
        'max_size': 256,
        'annotation_type': 'raycast',
        'patching_overlap_pct': 0.10,
    }
    chunker = SpatialChunker(config)

    # 512x512 image → should produce multiple chunks
    img = _make_image(512, 512)
    anns = _make_annotations(50, cx_range=(50, 462), cy_range=(50, 462), ray_value=30.0)

    for chunk_id, img_chunk, ann_chunk, tissue_val in chunker.process_roi('test_roi', img, anns, 1):
        chunk_h, chunk_w = img_chunk.shape[:2]

        if len(ann_chunk) > 0:
            cx = ann_chunk[:, CX_IDX]
            cy = ann_chunk[:, CY_IDX]
            assert np.all(cx >= 0) and np.all(cx < chunk_w), (
                f'Centroid x out of bounds in {chunk_id}: cx in [{cx.min()}, {cx.max()}), chunk_w={chunk_w}'
            )
            assert np.all(cy >= 0) and np.all(cy < chunk_h), (
                f'Centroid y out of bounds in {chunk_id}: cy in [{cy.min()}, {cy.max()}), chunk_h={chunk_h}'
            )

    print('PASS: SpatialChunker raycast branch — all centroids within chunk bounds')


# ---------------------------------------------------------------------------
# Test 5: SpatialChunker raycast branch — small image passes through unchanged
# ---------------------------------------------------------------------------


def test_spatial_chunker_raycast_small_image():
    config = {
        'max_size': 1024,
        'annotation_type': 'raycast',
        'patching_overlap_pct': 0.10,
    }
    chunker = SpatialChunker(config)

    img = _make_image(512, 512)
    anns = _make_annotations(10)
    chunks = list(chunker.process_roi('small', img, anns, 1))

    assert len(chunks) == 1, f'Expected 1 chunk for small image, got {len(chunks)}'
    chunk_id, img_out, anns_out, tissue_out = chunks[0]
    assert chunk_id == 'small'
    assert np.array_equal(img_out, img), 'Image should be unchanged for small ROI'
    assert np.array_equal(anns_out, anns), 'Annotations should be unchanged for small ROI'
    print('PASS: SpatialChunker raycast branch — small image passes through unchanged')


# ---------------------------------------------------------------------------
# Test 6: filter_and_clip_annotations — survival rate filtering works
# ---------------------------------------------------------------------------


def test_filter_survival_rate():
    # Cell at corner with large rays → most rays clipped to 0 after boundary clipping
    anns = np.zeros((1, 35), dtype=np.float32)
    anns[0, 0] = 1  # class
    anns[0, CX_IDX] = 1.0  # very close to left edge
    anns[0, CY_IDX] = 1.0  # very close to top edge
    anns[0, RAY_START_IDX:] = 100.0  # large rays → many will be clipped

    # Clip to a 100x100 region starting at (0,0)
    result = filter_and_clip_annotations(anns, 0, 0, 100, 100, min_rays_after_clip=0.5)

    # With centroid at (1,1) and rays of 100, many rays point left/up and get clipped to ~0
    # The cell should be dropped if < 50% of rays survive
    surviving_rays = result[:, RAY_START_IDX:]
    n_surviving = np.sum(surviving_rays > 0, axis=1)
    if len(result) > 0:
        assert np.all(n_surviving / 32 >= 0.5), 'Surviving cells must have >= 50% rays'
    print('PASS: filter_and_clip_annotations survival rate filtering works')


# ---------------------------------------------------------------------------
# Test 7: TransformOrchestrator end-to-end pipeline
# ---------------------------------------------------------------------------


def test_transform_orchestrator_e2e():
    """Full pipeline: write synthetic ingested .npz, run orchestrator, verify output."""
    with tempfile.TemporaryDirectory() as tmp:
        ingested_dir = Path(tmp) / 'ingested'
        output_dir = Path(tmp) / 'output'

        # Create synthetic ingested data
        dataset_dir = ingested_dir / 'test_dataset' / 'train'
        dataset_dir.mkdir(parents=True)

        img = _make_image(512, 512)
        anns = _make_annotations(10)
        np.savez_compressed(
            dataset_dir / 'roi_001.npz',
            image=img,
            annotations=anns,
            tissue=np.int32(1),
        )

        # Mock config manager
        class MockConfig:
            def get_global_config(self):
                return {
                    'max_size': 256,
                    'annotation_type': 'raycast',
                    'patching_overlap_pct': 0.10,
                }

        orchestrator = TransformOrchestrator(MockConfig(), str(ingested_dir), str(output_dir))
        orchestrator.run_pipeline()

        # Verify output files exist
        output_files = list(output_dir.glob('*.npz'))
        assert len(output_files) > 0, 'No output files produced'

        for npz_path in output_files:
            data = np.load(npz_path)

            # Check required keys
            assert 'image' in data, f'Missing image key in {npz_path.name}'
            assert 'annotations' in data, f'Missing annotations key in {npz_path.name}'
            assert 'tissue' in data, f'Missing tissue key in {npz_path.name}'
            assert 'content_h' in data, f'Missing content_h key in {npz_path.name}'
            assert 'content_w' in data, f'Missing content_w key in {npz_path.name}'

            content_h = int(data['content_h'])
            content_w = int(data['content_w'])
            img_out = data['image']

            # content_h/w <= target_size (256)
            assert content_h <= 256, f'content_h ({content_h}) > target_size (256)'
            assert content_w <= 256, f'content_w ({content_w}) > target_size (256)'

            # Unpadded tiles: content_h == image.shape[0]
            if content_h < 256:
                assert content_h == img_out.shape[0], f'Unpadded tile content_h mismatch'
            if content_w < 256:
                assert content_w == img_out.shape[1], f'Unpadded tile content_w mismatch'

            # Padded image is 256x256
            assert img_out.shape[0] == 256, f'Output image height should be 256, got {img_out.shape[0]}'
            assert img_out.shape[1] == 256, f'Output image width should be 256, got {img_out.shape[1]}'

            # Annotations are float32 (N, 35)
            anns_out = data['annotations']
            assert anns_out.dtype == np.float32, f'Annotations should be float32, got {anns_out.dtype}'
            if len(anns_out) > 0:
                assert anns_out.shape[1] == 35, f'Annotations should have 35 columns, got {anns_out.shape[1]}'

        print(f'PASS: TransformOrchestrator e2e — {len(output_files)} output tiles verified')


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)

    test_normalizer_returns_4_values()
    test_content_dims_before_padding()
    test_unpadded_content_dims()
    test_spatial_chunker_raycast_centroid_in_bounds()
    test_spatial_chunker_raycast_small_image()
    test_filter_survival_rate()
    test_transform_orchestrator_e2e()

    print('\nAll Phase 2 tests passed!')
