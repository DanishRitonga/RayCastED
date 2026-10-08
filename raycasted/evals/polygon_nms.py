"""Class-agnostic NMS for radial RayCast polygons.

This module is used only by the explicit duplicate-training experiment.  The
normal one-to-one configuration does not call it.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def raycast_vertices(polygons: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, rays...]`` polygons to ordered XY vertices."""
    polygons = np.asarray(polygons, dtype=np.float64)
    if polygons.ndim != 2 or polygons.shape[1] < 5:
        raise ValueError('polygons must have shape [N, 2 + n_rays]')
    n_rays = polygons.shape[1] - 2
    angles = np.arange(n_rays, dtype=np.float64) * (2.0 * np.pi / n_rays)
    rays = polygons[:, 2:]
    x = polygons[:, 0:1] + rays * np.cos(angles)[None]
    y = polygons[:, 1:2] + rays * np.sin(angles)[None]
    return np.stack((x, y), axis=-1)


def polygon_nms_indices(
    polygons: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    class_ids: np.ndarray | None = None,
    class_agnostic: bool = True,
) -> np.ndarray:
    """Return score-ordered indices after exact polygon-IoU NMS.

    Class-agnostic suppression is the duplicate experiment's intended
    inference contract: two queries describing the same nucleus compete even
    when their type predictions disagree.
    """
    polygons = np.asarray(polygons)
    scores = np.asarray(scores).reshape(-1)
    if len(polygons) != len(scores):
        raise ValueError('polygons and scores must have the same length')
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError('iou_threshold must be in [0, 1]')
    if len(polygons) == 0:
        return np.empty(0, dtype=np.int64)

    classes = None if class_ids is None else np.asarray(class_ids).reshape(-1)
    if not class_agnostic and (classes is None or len(classes) != len(polygons)):
        raise ValueError('class_ids must match polygons for class-aware NMS')

    geometries = []
    for vertices in raycast_vertices(polygons):
        geometry = Polygon(vertices)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        geometries.append(geometry)

    order = np.argsort(-scores, kind='stable')
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    tree = STRtree(geometries)
    suppressed = np.zeros(len(polygons), dtype=bool)
    keep = []
    for pos, idx in enumerate(order):
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        geom_i = geometries[idx]
        if geom_i.is_empty or geom_i.area <= 0:
            continue
        candidates = tree.query(geom_i)
        candidates = candidates[rank[candidates] > pos]
        candidates = candidates[np.argsort(rank[candidates], kind='stable')]
        for other in candidates:
            if suppressed[other] or (not class_agnostic and classes[other] != classes[idx]):
                continue
            geom_o = geometries[other]
            if geom_o.is_empty or geom_o.area <= 0:
                continue
            intersection = geom_i.intersection(geom_o).area
            if intersection <= 0:
                continue
            union = geom_i.area + geom_o.area - intersection
            if union > 0 and intersection / union > iou_threshold:
                suppressed[other] = True
    return np.asarray(keep, dtype=np.int64)
