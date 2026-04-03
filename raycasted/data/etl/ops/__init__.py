"""Polygon YOLOv26 — Operations Module

Single source of truth for all polygon/raycast geometry logic.
This module is imported by:
    - raycasted/data/etl/ingestors/*.py (NumPy, offline ETL)
    - raycasted/data/etl/transform/*.py (NumPy, offline ETL)
    - raycasted/data/etl/loader/polygon_dataset.py (NumPy, online DataLoader)
    - ultralytics/utils/loss.py (PyTorch, training)
    - ultralytics/utils/tal.py (PyTorch, assignment)
    - ultralytics/models/yolo/detect/predict.py (PyTorch, inference)

PyTorch functions use LAZY IMPORTS (import torch inside function body)
so this module can be imported in ETL environments without PyTorch installed.

Module Structure:
    - convert:    Polygon ↔ raycast conversion
    - filter:     Annotation filtering and clipping
    - iou:        Polar-IoU computation (NumPy + PyTorch)
    - loss:       Angular smoothness regularization (NumPy + PyTorch)
    - augment:    Geometric augmentations
    - utils:      Validation and quality monitoring
"""

from .augment import (
    flip_horizontal,
    flip_vertical,
    rotate_90,
)
from .convert import (
    decode_to_vertices,
    polygon_to_raycast,
    raycast_to_annotation,
    raycast_to_polygon,
)
from .filter import filter_and_clip_annotations
from .iou import (
    polar_iou,
    polar_iou_pairwise_flat,
    polar_iou_pairwise_flat_torch,
    polar_iou_torch,
)
from .loss import (
    angular_smoothness_loss,
    angular_smoothness_loss_torch,
)
from .utils import (
    count_zero_rays,
    validate_annotation_format,
)

__all__ = [
    # Convert
    'polygon_to_raycast',
    'raycast_to_annotation',
    'raycast_to_polygon',
    'decode_to_vertices',
    # Filter
    'filter_and_clip_annotations',
    # IoU
    'polar_iou',
    'polar_iou_pairwise_flat',
    'polar_iou_torch',
    'polar_iou_pairwise_flat_torch',
    # Loss
    'angular_smoothness_loss',
    'angular_smoothness_loss_torch',
    # Augment
    'flip_horizontal',
    'flip_vertical',
    'rotate_90',
    # Utils
    'count_zero_rays',
    'validate_annotation_format',
]
