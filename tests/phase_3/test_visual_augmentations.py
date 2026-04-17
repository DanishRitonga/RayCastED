"""Phase 3 visual test — individual augmentation operations.

Creates a (4, N+1) grid:
  - Column 0: original annotation
  - Column 1..N: result of each augmentation applied independently

Run with: uv run python tests/phase_3/test_visual_augmentations.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.affinity import rotate as shapely_rotate, scale as shapely_scale, translate as shapely_translate
from shapely.geometry import Point

from raycasted.data.etl.ops.augment import flip_horizontal, flip_vertical, rotate_90
from raycasted.data.etl.ops.convert import decode_to_vertices, polygon_to_raycast
from raycasted.data.etl.utils.constants import CX_IDX, CY_IDX, RAY_END_IDX, RAY_START_IDX


CROP = 400

# Augmentation operations: (label, callable)
AUGMENTATIONS = [
    ('Original', None),
    ('H-Flip', 'flip_h'),
    ('V-Flip', 'flip_v'),
    ('Rot 90°', 'rot1'),
    ('Rot 180°', 'rot2'),
    ('Rot 270°', 'rot3'),
]


def _make_oval(cx, cy, semi_a, semi_b, angle_deg):
    """Create a rotated oval polygon."""
    circle = Point(0, 0).buffer(1.0, resolution=32)
    oval = shapely_scale(circle, xfact=semi_a, yfact=semi_b)
    oval = shapely_rotate(oval, angle_deg, origin=(0, 0))
    oval = shapely_translate(oval, xoff=cx, yoff=cy)
    return oval


def _make_image_with_ovals(crop, ovals, bg_rng):
    """Create a synthetic image with coloured patches for each oval."""
    img = np.zeros((crop, crop, 3), dtype=np.uint8)
    # Warm pinkish background (H&E-ish)
    img[:, :, 0] = bg_rng.integers(180, 220, (crop, crop), dtype=np.uint8)
    img[:, :, 1] = bg_rng.integers(140, 180, (crop, crop), dtype=np.uint8)
    img[:, :, 2] = bg_rng.integers(160, 200, (crop, crop), dtype=np.uint8)

    # Paint each oval with a distinct colour patch
    for oval in ovals:
        minx, miny, maxx, maxy = [int(v) for v in oval.bounds]
        minx, miny = max(0, minx), max(0, miny)
        maxx, maxy = min(crop, maxx), min(crop, maxy)
        color = bg_rng.integers(60, 140, 3, dtype=np.uint8)
        img[miny:maxy, minx:maxx] = color

    return img


def _overlay_polygon(ax, ann, color, linewidth=1.5):
    """Draw a single polygon annotation on axes."""
    cx = ann[CX_IDX]
    cy = ann[CY_IDX]
    rays = ann[RAY_START_IDX:RAY_END_IDX]
    verts = decode_to_vertices(rays[np.newaxis], np.array([cx]), np.array([cy]))

    vx = np.append(verts[0, :, 0], verts[0, 0, 0])
    vy = np.append(verts[0, :, 1], verts[0, 0, 1])
    ax.plot(vx, vy, color=color, linewidth=linewidth, alpha=0.9)
    ax.plot(cx, cy, 'w+', markersize=8, markeredgewidth=1.5)


def _augment_ann(ann, op):
    """Apply a single augmentation operation to annotations."""
    if op is None:
        return ann.copy()
    elif op == 'flip_h':
        return flip_horizontal(ann[np.newaxis], CROP)[0]
    elif op == 'flip_v':
        return flip_vertical(ann[np.newaxis], CROP)[0]
    elif op == 'rot1':
        return rotate_90(ann[np.newaxis], 1, CROP)[0]
    elif op == 'rot2':
        return rotate_90(ann[np.newaxis], 2, CROP)[0]
    elif op == 'rot3':
        return rotate_90(ann[np.newaxis], 3, CROP)[0]


def _augment_img(img, op):
    """Apply a single augmentation to an image."""
    if op is None:
        return img.copy()
    elif op == 'flip_h':
        return img[:, ::-1, :]
    elif op == 'flip_v':
        return img[::-1, :, :]
    elif op == 'rot1':
        return np.rot90(img, k=1)
    elif op == 'rot2':
        return np.rot90(img, k=2)
    elif op == 'rot3':
        return np.rot90(img, k=3)


def generate_augmentation_visual():
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    bg_rng = np.random.default_rng(42)

    # 4 rows, each with a different oval shape and position
    oval_specs = [
        # (cx, cy, semi_a, semi_b, angle_deg) — vary eccentricity and rotation
        (200, 200, 60, 15, 0),
        (250, 180, 45, 12, 45),
        (180, 220, 50, 18, 120),
        (220, 200, 35, 10, 75),
    ]

    ncols = len(AUGMENTATIONS)
    nrows = len(oval_specs)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    fig.suptitle('Augmentation Operations — each column is one isolated transform', fontsize=14)

    for row_idx, (cx, cy, sa, sb, angle) in enumerate(oval_specs):
        oval = _make_oval(cx, cy, sa, sb, angle)
        ann = polygon_to_raycast(oval, class_id=1)
        assert ann is not None, f'polygon_to_raycast returned None for oval {row_idx + 1}'

        # Build a simple image with a coloured patch for this oval
        img = _make_image_with_ovals(CROP, [oval], bg_rng)

        for col_idx, (label, op) in enumerate(AUGMENTATIONS):
            ax = axes[row_idx, col_idx]
            aug_img = _augment_img(img, op)
            aug_ann = _augment_ann(ann, op)

            ax.imshow(aug_img)

            # Draw original polygon outline (ghosted) for comparison
            _overlay_polygon(ax, ann, color='white', linewidth=0.5)

            # Draw augmented polygon
            _overlay_polygon(ax, aug_ann, color='lime', linewidth=1.8)

            ax.set_xlim(0, CROP)
            ax.set_ylim(CROP, 0)

            if row_idx == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'Oval {row_idx + 1}\n(a={sa}, b={sb}, θ={angle}°)', fontsize=9)

            # Show centroid coords for verification
            ax.text(
                4,
                CROP - 8,
                f'cx={aug_ann[CX_IDX]:.0f} cy={aug_ann[CY_IDX]:.0f}',
                color='yellow',
                fontsize=7,
                va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, pad=1),
            )

            ax.tick_params(labelsize=5)

    # Legend
    fig.text(
        0.5,
        0.01,
        'white = original polygon  |  green = augmented polygon  |  + = centroid',
        ha='center',
        fontsize=10,
        style='italic',
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = output_dir / 'augmentations.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[INFO] saved -> {out}')


if __name__ == '__main__':
    generate_augmentation_visual()
