"""RayCastED — Conversion Operations

Core conversion functions between polygon and raycast representations.
These are used during ETL ingestion and inference decoding.
"""

import collections
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from ..utils.constants import (
    CLASS_IDX,
    CX_IDX,
    CY_IDX,
    N_RAYS,
    RAY_ANGLES,
    RAY_COS,
    RAY_END_IDX,
    RAY_SIN,
    RAY_START_IDX,
)


def polygon_to_raycast(
    poly: Polygon,
    class_id: int,
    n_rays: int = 32,
    fallback_counter: collections.Counter | None = None,
) -> np.ndarray | None:
    """Convert a Shapely polygon to a raycast annotation row.

    Returns:
        np.ndarray of shape (35,) in unified format:
            [class_id, cx, cy, d_1, ..., d_32]  — pixel space
        None if the polygon has zero area or is unrecoverable.

    Centroid policy:
        1. Try area-weighted Shapely centroid.
        2. If centroid is outside polygon: fall back to representative_point().
        3. Update cx, cy to whichever point is used for ray casting.
           Never cast rays from one point and store a different centroid.

    Args:
        poly: Shapely Polygon to convert.
        class_id: Integer class label for the annotation.
        n_rays: Number of radial rays (default 32).
        fallback_counter: Optional Counter for diagnostic logging.
            Increments key 'representative_point_fallback' when triggered.
    """
    import shapely

    if poly is None or poly.is_empty or poly.area == 0:
        return None

    poly = poly.buffer(0)  # heal self-intersections
    if not poly.is_valid or poly.is_empty:
        return None

    # --- Centroid selection ---
    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y

    if not poly.contains(Point(cx, cy)):
        rep = poly.representative_point()
        cx = rep.x  # UPDATE cx/cy before ray casting
        cy = rep.y
        if fallback_counter is not None:
            fallback_counter['representative_point_fallback'] += 1

    # --- R_far: bounding-box diagonal + 10% ---
    minx, miny, maxx, maxy = poly.bounds
    bbox_w = maxx - minx
    bbox_h = maxy - miny
    R_far = math.sqrt(bbox_w ** 2 + bbox_h ** 2) * 1.1

    # --- Ray casting ---
    shapely.prepare(poly)  # build spatial index once; ~5x faster per-ray query
    rays = np.zeros(n_rays, dtype=np.float32)

    angles = RAY_ANGLES  # from raycasted.data.etl.utils.constants

    for i, theta in enumerate(angles):
        dx = math.cos(theta) * R_far
        dy = math.sin(theta) * R_far
        ray_line = LineString([(cx, cy), (cx + dx, cy + dy)])

        intersection = poly.boundary.intersection(ray_line)
        if intersection.is_empty:
            rays[i] = 0.0
            continue

        nearest = nearest_points(Point(cx, cy), intersection)[1]
        ix, iy = nearest.x, nearest.y

        rays[i] = math.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)

    return raycast_to_annotation(cx, cy, rays, class_id)


def raycast_to_annotation(
    cx: float,
    cy: float,
    rays: np.ndarray,
    class_id: int,
) -> np.ndarray:
    """Package raycast data into unified format array.

    Args:
        cx: Centroid x-coordinate
        cy: Centroid y-coordinate
        rays: Array of 32 ray distances
        class_id: Class ID

    Returns:
        annotation: Array of shape (35,) — [class_id, cx, cy, d_1, ..., d_32]
    """
    annotation = np.zeros(35, dtype=np.float32)
    annotation[CLASS_IDX] = float(class_id)
    annotation[CX_IDX] = cx
    annotation[CY_IDX] = cy
    annotation[RAY_START_IDX:RAY_END_IDX] = rays
    return annotation


def raycast_to_polygon(
    rays: np.ndarray,
    cx: float,
    cy: float,
) -> Polygon:
    """Convert raycast representation to a Shapely Polygon.

    Args:
        rays: Ray distances, shape (32,)
        cx: Centroid x-coordinate
        cy: Centroid y-coordinate

    Returns:
        polygon: Shapely Polygon object
    """
    rays = np.asarray(rays, dtype=np.float64)

    vertex_x = cx + rays * RAY_COS
    vertex_y = cy + rays * RAY_SIN

    coords = list(zip(vertex_x, vertex_y))

    return Polygon(coords)


def decode_to_vertices(
    rays: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> np.ndarray:
    """Decode ray distances to Cartesian polygon vertices.

    Args:
        rays: Ray distances, shape (N, 32) in pixel space
        cx: Centroid x-coordinates, shape (N,)
        cy: Centroid y-coordinates, shape (N,)

    Returns:
        vertices: Polygon vertices, shape (N, 32, 2) in pixel space
            vertices[i, j, 0] = x-coordinate of j-th vertex of i-th polygon
            vertices[i, j, 1] = y-coordinate of j-th vertex of i-th polygon
    """
    rays = np.asarray(rays, dtype=np.float64)
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)

    vertex_x = cx[:, np.newaxis] + rays * RAY_COS[np.newaxis, :]
    vertex_y = cy[:, np.newaxis] + rays * RAY_SIN[np.newaxis, :]

    vertices = np.stack([vertex_x, vertex_y], axis=2)

    return vertices.astype(np.float32)
