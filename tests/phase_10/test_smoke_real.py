"""Phase 10 real-data smoke test — PUMA dataset.

Converts real PUMA GeoJSON nuclei annotations to raycast format,
saves as .npz tiles, Runs 2-epoch training on CPU to validate the full
pipeline end-to-end with real histopathology data.

Usage:
    uv run python tests/phase_10/test_smoke_real.py
"""

import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon

from raycasted.data.etl.ops.convert import polygon_to_raycast
from raycasted.model.head import RayCastDetect
from raycasted.model.train import RayCastTrainer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PUMA_ROOT = Path('/home/danishrtg/projects/RayCastED/raycasted/data/dataset/PUMA')
IMG_DIR = PUMA_ROOT / '01_training_dataset_tif_ROIs'
ANN_DIR = PUMA_ROOT / '01_training_dataset_geojson_nuclei'

CROP_SIZE = 640
N_ROIS = 5  # Use 5 ROIs (4 train, 1 val)
N_RAYS = 32

# PUMA class mapping (from dataset.yaml namespace_map)
PUMA_CLASS_MAP = {
    'nuclei_tumor': 0,
    'nuclei_lymphocyte': 1,
    'nuclei_plasma_cell': 2,
    'nuclei_histiocyte': 2,
    'nuclei_melanophage': 2,
    'nuclei_neutrophil': 2,
    'nuclei_stroma': 3,
    'nuclei_endothelium': 3,
    'nuclei_epithelium': 0,
    'nuclei_apoptosis': 255,  # ignore
}

GLOBAL_CLASSES = {0: 'Epithelial', 1: 'Lymphocyte', 2: 'Immune Cells', 3: 'Stroma'}
NC = len(GLOBAL_CLASSES)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _load_roi(roi_stem: str) -> tuple[np.ndarray, list[dict]]:
    """Load image and parse GeoJSON annotations for one ROI."""
    img_path = IMG_DIR / f'{roi_stem}.tif'
    ann_path = ANN_DIR / f'{roi_stem}_nuclei.geojson'

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f'Cannot load {img_path}')

    with open(ann_path) as f:
        geojson = json.load(f)

    features = geojson['features']
    return img, features


def _geojson_to_raycast_annotations(features: list[dict], class_map: dict) -> np.ndarray:
    """Convert GeoJSON polygon features to raycast annotation rows [N, 35]."""
    rows = []
    for feat in features:
        props = feat.get('properties', {})
        cls_name = props.get('classification', {}).get('name', '')
        class_id = class_map.get(cls_name, 255)
        if class_id == 255:
            continue

        coords = feat['geometry']['coordinates'][0]
        if len(coords) < 3:
            continue

        poly = Polygon(coords)
        if not poly.is_valid or poly.area == 0:
            poly = poly.buffer(0)
            if not poly.is_valid or poly.area == 0:
                continue

        row = polygon_to_raycast(poly, class_id, n_rays=N_RAYS)
        if row is not None:
            rows.append(row)

    if rows:
        return np.stack(rows)
    return np.zeros((0, 35), dtype=np.float32)


def _save_roi_as_tile(
    img: np.ndarray,
    annotations: np.ndarray,
    output_path: str,
    crop_size: int = 640,
) -> int:
    """Resize image to crop_size, adjust annotations. Save as .npz tile."""
    h, w = img.shape[:2]

    # Resize image
    img_resized = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    # Scale annotations
    scale_x = crop_size / w
    scale_y = crop_size / h

    if annotations.shape[0] > 0:
        scaled = annotations.copy()
        scaled[:, 1] *= scale_x  # cx
        scaled[:, 2] *= scale_y  # cy
        scaled[:, 3:] *= min(scale_x, scale_y)  # rays
    else:
        scaled = annotations

    np.savez_compressed(
        output_path,
        image=img_resized,
        annotations=scaled,
        tissue=np.int32(1),
        content_h=np.int32(crop_size),
        content_w=np.int32(crop_size),
    )
    return scaled.shape[0]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_smoke_real_puma():
    """Smoke test: train 2 epochs on real PUMA data with CPU."""
    start = time.time()

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_dir = os.path.join(tmp_dir, 'train')
        val_dir = os.path.join(tmp_dir, 'val')
        os.makedirs(train_dir)
        os.makedirs(val_dir)

        # Get ROI stems sorted by name
        roi_stems = sorted([p.stem for p in IMG_DIR.glob('*.tif')])[:N_ROIS]
        train_stems = roi_stems[:4]
        val_stems = roi_stems[4:]

        print(f'Converting {len(train_stems)} train ROIs + {len(val_stems)} val ROIs ...')

        total_annots = 0
        for i, stem in enumerate(train_stems):
            img, features = _load_roi(stem)
            anns = _geojson_to_raycast_annotations(features, PUMA_CLASS_MAP)
            tile_path = os.path.join(train_dir, f'train_{i:04d}.npz')
            n = _save_roi_as_tile(img, anns, tile_path, CROP_SIZE)
            total_annots += n
            print(f'  [{stem}] {anns.shape[0]} annotations → saved')

        for i, stem in enumerate(val_stems):
            img, features = _load_roi(stem)
            anns = _geojson_to_raycast_annotations(features, PUMA_CLASS_MAP)
            tile_path = os.path.join(val_dir, f'val_{i:04d}.npz')
            n = _save_roi_as_tile(img, anns, tile_path, CROP_SIZE)
            total_annots += n
            print(f'  [val: {stem}] {anns.shape[0]} annotations → saved')

        print(f'Total: {total_annots} annotations across {len(train_stems) + len(val_stems)} ROIs')

        # Create YAML
        yaml_data = {
            'train': train_dir,
            'val': val_dir,
            'nc': NC,
            'names': GLOBAL_CLASSES,
        }
        yaml_path = os.path.join(tmp_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)

        # Run training
        print(f'\nStarting training on {NC} classes, crop={CROP_SIZE} ...')
        trainer = RayCastTrainer(
            overrides={
                'model': 'yolo11n.yaml',
                'data': yaml_path,
                'epochs': 2,
                'batch': 4,
                'imgsz': CROP_SIZE,
                'mosaic': 0.0,
                'mixup': 0.0,
                'cache': False,
                'pretrained': False,
                'device': 'cpu',
                'workers': 0,
                'verbose': False,
                'plots': False,
                'save': False,
                'val': True,
            }
        )

        trainer.train()
        elapsed = time.time() - start

        # Verify
        model = trainer.model
        head = model.model[-1]
        assert isinstance(head, RayCastDetect), f'Expected RayCastDetect, got {type(head).__name__}'
        ta = model.training_args
        assert ta['nc'] == NC
        assert ta['n_rays'] == 32

        print(f'\nPASS: real-data smoke train — {NC} classes, 2 epochs in {elapsed:.1f}s')
        print(f'  Model: YOLO11n + RayCastDetect (nc={NC}, n_rays=32)')
        print(f'  Annotations: {total_annots}')


if __name__ == '__main__':
    test_smoke_real_puma()
