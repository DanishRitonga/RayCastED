"""Phase 3 visual stress-test — oval-shaped annotations with random rotation.

Generates multiple grids of augmented crops to verify ray clipping,
augmentation, and polygon overlay correctness for asymmetric shapes.
Run with: uv run python tests/phase_3/test_visual_ovals.py
"""

import math
import tempfile
from pathlib import Path

import numpy as np

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.ops.convert import decode_to_vertices, polygon_to_raycast
from raycasted.data.etl.utils.constants import CX_IDX, CY_IDX, RAY_END_IDX, RAY_START_IDX

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.affinity import rotate
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_oval_polygon(cx: float, cy: float, semi_a: float, semi_b: float, angle_deg: float) -> 'Polygon':
    """Create a rotated oval (ellipse) as a Shapely Polygon."""
    circle = Point(0, 0).buffer(1.0, resolution=32)
    from shapely.affinity import scale
    oval = scale(circle, xfact=semi_a, yfact=semi_b)
    oval = rotate(oval, angle_deg, origin=(0, 0))
    from shapely.affinity import translate
    oval = translate(oval, xoff=cx, yoff=cy)
    return oval


def _make_oval_annotations(
    n_cells: int,
    canvas_h: int,
    canvas_w: int,
    rng: np.random.Generator,
    semi_a_range=(10, 40),
    semi_b_range=(5, 15),
) -> np.ndarray:
    """Create raycast annotations from randomly rotated ovals."""
    annotations = []
    margin = 50
    for _ in range(n_cells):
        cx = rng.uniform(margin, canvas_w - margin)
        cy = rng.uniform(margin, canvas_h - margin)
        semi_a = rng.uniform(*semi_a_range)
        semi_b = rng.uniform(*semi_b_range)
        angle = rng.uniform(0, 360)

        oval = _make_oval_polygon(cx, cy, semi_a, semi_b, angle)
        row = polygon_to_raycast(oval, class_id=1)
        if row is not None:
            annotations.append(row)

    if not annotations:
        return np.zeros((0, 35), dtype=np.float32)
    return np.stack(annotations)


def _make_image(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    img = rng.integers(80, 200, size=(h, w, 3), dtype=np.uint8)
    img[:, :, 0] = rng.integers(150, 220, size=(h, w), dtype=np.uint8)
    return img


def _create_tile_with_ovals(
    directory: Path,
    name: str,
    image_size: tuple[int, int],
    content_h: int | None = None,
    content_w: int | None = None,
    n_cells: int = 20,
    rng: np.random.Generator | None = None,
    semi_a_range=(12, 45),
    semi_b_range=(5, 16),
) -> Path:
    """Write a synthetic .npz tile with oval annotations."""
    if rng is None:
        rng = np.random.default_rng()
    h, w = image_size
    img = _make_image(h, w)
    anns = _make_oval_annotations(n_cells, h, w, rng, semi_a_range, semi_b_range)

    ch = content_h if content_h is not None else h
    cw = content_w if content_w is not None else w

    path = directory / name
    np.savez_compressed(path, image=img, annotations=anns, tissue=np.int32(1),
                        content_h=np.int32(ch), content_w=np.int32(cw))
    return path


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_crops(ds, ax_idx_offset, axes_flat, crop_size):
    """Plot augmented crops on the given axes."""
    from shapely.geometry import Polygon as ShapelyPolygon

    for i, ax in enumerate(axes_flat):
        img_tensor, labels = ds[0]

        img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        ax.imshow(img_np)

        n_cells = len(labels)
        if n_cells > 0:
            pixel_labels = labels.copy()
            pixel_labels[:, CX_IDX] *= crop_size
            pixel_labels[:, CY_IDX] *= crop_size
            pixel_labels[:, RAY_START_IDX:RAY_END_IDX] *= crop_size

            cx = pixel_labels[:, CX_IDX]
            cy = pixel_labels[:, CY_IDX]
            rays = pixel_labels[:, RAY_START_IDX:RAY_END_IDX]
            vertices = decode_to_vertices(rays, cx, cy)

            for j in range(n_cells):
                vx = np.append(vertices[j, :, 0], vertices[j, 0, 0])
                vy = np.append(vertices[j, :, 1], vertices[j, 0, 1])

                # Color by proximity to edge (cyan=safe, red=clipped)
                dist_to_edge = min(cx[j], cy[j], crop_size - cx[j], crop_size - cy[j])
                color = 'red' if dist_to_edge < 30 else ('yellow' if dist_to_edge < 60 else 'cyan')
                ax.plot(vx, vy, color=color, linewidth=1.0, alpha=0.85)
                ax.plot(cx[j], cy[j], 'r+', markersize=3)

        ax.set_xlim(0, crop_size)
        ax.set_ylim(crop_size, 0)
        panel_num = ax_idx_offset + i + 1
        edge_info = f', near-edge in red' if n_cells > 0 else ''
        ax.set_title(f'Panel {panel_num}: {n_cells} cells{edge_info}', fontsize=9)
        ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Main visual generation
# ---------------------------------------------------------------------------

def generate_oval_visuals():
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_size = 640
    rng = np.random.default_rng(123)

    # === Page 1: Dense ovals, many cells, lots of edge proximity ===
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        # Tile larger than crop → random crops will place cells near edges
        _create_tile_with_ovals(
            tile_dir, name='dense.npz',
            image_size=(1200, 1200), content_h=1100, content_w=1100,
            n_cells=40, rng=rng,
            semi_a_range=(15, 50), semi_b_range=(5, 18),
        )

        ds = RayCastTileDataset(str(tile_dir), crop_size=crop_size, augment=True)

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Oval Annotations — Dense Edge Cases (cyan=safe, yellow=close, red=near-edge)',
                      fontsize=13)
        _plot_crops(ds, 0, axes.flat, crop_size)
        plt.tight_layout()
        out = output_dir / 'oval_dense.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'[INFO] Page 1 saved -> {out}')

    # === Page 2: Fewer large ovals, spread to catch clipping artifacts ===
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        _create_tile_with_ovals(
            tile_dir, name='large.npz',
            image_size=(1024, 1024), content_h=950, content_w=950,
            n_cells=12, rng=rng,
            semi_a_range=(30, 60), semi_b_range=(8, 20),
        )

        ds = RayCastTileDataset(str(tile_dir), crop_size=crop_size, augment=True)

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Oval Annotations — Large Shapes (cyan=safe, yellow=close, red=near-edge)',
                      fontsize=13)
        _plot_crops(ds, 12, axes.flat, crop_size)
        plt.tight_layout()
        out = output_dir / 'oval_large.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'[INFO] Page 2 saved -> {out}')

    # === Page 3: No augmentation — verify raw crop clipping only ===
    with tempfile.TemporaryDirectory() as tmp:
        tile_dir = Path(tmp)
        _create_tile_with_ovals(
            tile_dir, name='noaug.npz',
            image_size=(1200, 1200), content_h=1100, content_w=1100,
            n_cells=30, rng=rng,
            semi_a_range=(15, 50), semi_b_range=(5, 18),
        )

        ds = RayCastTileDataset(str(tile_dir), crop_size=crop_size, augment=False)

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Oval Annotations — No Augmentation (raw crop clipping only)',
                      fontsize=13)
        _plot_crops(ds, 24, axes.flat, crop_size)
        plt.tight_layout()
        out = output_dir / 'oval_noaug.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'[INFO] Page 3 saved -> {out}')


if __name__ == '__main__':
    generate_oval_visuals()
    print('\nDone — check tests/phase_3/output/ for oval_*.png')
