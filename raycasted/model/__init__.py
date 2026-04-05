"""RayCastED — Model package.

Provides RayCastDetect head and registration utilities for training
raycast polygon detectors with the Ultralytics YOLO framework.
"""

from .head import RayCastDetect, RayRefinementBlock
from .loss import RayCastDetectionLoss, RayCastE2ELoss
from .register import register_raycast_head

__all__ = ['RayCastDetect', 'RayRefinementBlock', 'register_raycast_head', 'RayCastDetectionLoss', 'RayCastE2ELoss']
