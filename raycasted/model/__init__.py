"""RayCastED — Model package.

Provides PolygonDetect head and registration utilities for training
raycast polygon detectors with the Ultralytics YOLO framework.
"""

from .head import PolygonDetect, RayRefinementBlock
from .register import register_polygon_head

__all__ = ['PolygonDetect', 'RayRefinementBlock', 'register_polygon_head']
