"""RayCastED — Model package.

Provides RayCastDetect head, loss functions, prediction pipeline,
and registration utilities for training raycast polygon detectors
with the Ultralytics YOLO framework.
"""

from .annotate import RayCastAnnotator
from .head import RayCastDetect, RayRefinementBlock
from .loss import RayCastDetectionLoss, RayCastE2ELoss
from .predict import RayCastPredictor
from .register import register_raycast_head
from .val import RayCastValidator

__all__ = [
    'RayCastAnnotator',
    'RayCastDetect',
    'RayCastDetectionLoss',
    'RayCastE2ELoss',
    'RayCastPredictor',
    'RayCastValidator',
    'RayRefinementBlock',
    'register_raycast_head',
]
