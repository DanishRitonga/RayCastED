"""Checks for the held-out PanNuke evaluation contract."""

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from raycasted.evals.eval_lsp import rasterize_polys


def _polygon(cx: float, cy: float, radius: float) -> np.ndarray:
    return np.r_[cx, cy, np.full(64, radius)]


def test_overlapping_polygons_use_lsp_distance_ownership() -> None:
    """Overlapping pixels follow the official per-mask interior distance rule."""
    polygons = np.stack([_polygon(36, 48, 23), _polygon(57, 48, 23)])
    labels = np.array([1, 4])
    masks, kept = rasterize_polys(polygons, 96, 96, labels)

    raw = np.zeros((2, 96, 96), dtype=bool)
    for i, polygon in enumerate(polygons):
        rays = polygon[2:]
        angles = np.arange(64) * (2 * np.pi / 64)
        xy = np.stack([polygon[0] + rays * np.cos(angles), polygon[1] + rays * np.sin(angles)], axis=-1)
        image = Image.new('1', (96, 96))
        ImageDraw.Draw(image).polygon(xy.flatten().tolist(), outline=1, fill=1)
        raw[i] = np.asarray(image, dtype=bool)
    distance = np.stack([distance_transform_edt(mask) for mask in raw])
    expected = (np.arange(2)[:, None, None] == distance.argmax(axis=0)) & (distance > 0)

    assert np.array_equal(kept, labels)
    assert np.array_equal(np.stack(masks), expected)
    assert not (masks[0] & masks[1]).any()


def test_empty_polygon_and_label_are_removed_together() -> None:
    """A swallowed prediction must not retain a class label."""
    polygons = np.stack([_polygon(45, 45, 24), _polygon(45, 45, 9)])
    masks, labels = rasterize_polys(polygons, 96, 96, np.array([2, 3]))

    assert len(masks) == 1
    assert labels.tolist() == [2]
