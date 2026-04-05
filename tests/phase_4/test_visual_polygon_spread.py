"""Phase 4 visual — Decoded polygon spread.

Visualises all 8400 decoded polygons from random logits across
the three scale levels (P3/P4/P5) to verify spatial coverage,
shape diversity, and scale-dependent ray distributions.

Run with: uv run python tests/phase_4/test_visual_polygon_spread.py
"""

import numpy as np
import torch

from raycasted.data.etl.ops.convert import decode_to_vertices
from raycasted.model.head import RayCastDetect

NC = 4
REG_MAX = 16
CH = (64, 128, 256)
FEAT_SIZE = 80
IMG_SIZE = FEAT_SIZE * 8  # 640


def _make_feats(batch=1, feat_size=FEAT_SIZE, ch=CH):
    return [torch.randn(batch, c, feat_size // (2**i), feat_size // (2**i)) for i, c in enumerate(ch)]


def generate_polygon_spread_plots():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    head = RayCastDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    head.eval()
    head.stride = torch.tensor([8.0, 16.0, 32.0])

    feats = _make_feats()
    with torch.no_grad():
        preds = head.forward_head(feats, box_head=head.cv2, cls_head=head.cv3)
        decoded = head._inference(preds)

    # decoded: [1, 38, 8400] — xy(2) + rays(32) + scores(4)
    xy = decoded[0, :2, :].T.numpy()  # [8400, 2]
    rays = decoded[0, 2:34, :].T.numpy()  # [8400, 32]
    scores = decoded[0, 34:, :].T.numpy()  # [8400, 4]

    # Decode all polygon vertices
    vertices = decode_to_vertices(rays, xy[:, 0], xy[:, 1])  # [8400, 32, 2]

    # Scale splits
    n_p3 = 80 * 80  # 6400
    n_p4 = 40 * 40  # 1600
    n_p5 = 20 * 20  # 400

    max_score = scores.max(axis=1)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Phase 4 — Decoded Polygon Spread', fontsize=14, fontweight='bold')

    # ---- Row 1: Full grid overview ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('All 8400 Centroids (coloured by scale)', fontsize=10)
    ax.scatter(xy[:n_p3, 0], xy[:n_p3, 1], s=0.3, alpha=0.4, c='steelblue', label='P3', rasterized=True)
    ax.scatter(
        xy[n_p3 : n_p3 + n_p4, 0],
        xy[n_p3 : n_p3 + n_p4, 1],
        s=1,
        alpha=0.5,
        c='darkorange',
        label='P4',
        rasterized=True,
    )
    ax.scatter(xy[n_p3 + n_p4 :, 0], xy[n_p3 + n_p4 :, 1], s=3, alpha=0.7, c='seagreen', label='P5', rasterized=True)
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(IMG_SIZE, 0)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.legend(fontsize=8, markerscale=10)
    ax.set_aspect('equal')

    # ---- Row 1: Ray mean per anchor ----
    ray_means = rays.mean(axis=1)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title('Mean Ray Length per Anchor', fontsize=10)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c=ray_means, cmap='viridis', alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Mean ray (px)', shrink=0.8)
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(IMG_SIZE, 0)
    ax.set_xlabel('X (px)')
    ax.set_aspect('equal')

    # ---- Row 1: Max class score per anchor ----
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title('Max Class Score per Anchor', fontsize=10)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c=max_score, cmap='magma', alpha=0.6, vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Max score', shrink=0.8)
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(IMG_SIZE, 0)
    ax.set_xlabel('X (px)')
    ax.set_aspect('equal')

    # ---- Row 2: Sample polygons per scale ----
    np.random.seed(42)
    sample_config = [
        ('P3 (stride=8)', 0, n_p3, 50, 'steelblue'),
        ('P4 (stride=16)', n_p3, n_p4, 30, 'darkorange'),
        ('P5 (stride=32)', n_p3 + n_p4, n_p5, 20, 'seagreen'),
    ]

    for col, (label, start, count, n_sample, color) in enumerate(sample_config):
        ax = fig.add_subplot(gs[1, col])
        ax.set_title(f'{label} — {n_sample} sampled polygons', fontsize=10)
        indices = np.random.choice(np.arange(start, start + count), size=min(n_sample, count), replace=False)

        for idx in indices:
            verts = vertices[idx]
            # Close the polygon
            closed = np.vstack([verts, verts[0]])
            alpha = 0.15 + 0.6 * max_score[idx]
            ax.plot(closed[:, 0], closed[:, 1], linewidth=0.5, alpha=alpha, color=color)
            ax.plot(xy[idx, 0], xy[idx, 1], '.', markersize=2, color=color, alpha=0.5)

        ax.set_xlim(0, IMG_SIZE)
        ax.set_ylim(IMG_SIZE, 0)
        ax.set_xlabel('X (px)')
        ax.set_aspect('equal')

    # ---- Row 3: Statistics panels ----
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title('Ray Length Distribution per Scale', fontsize=10)
    ax.hist(rays[:n_p3].flatten(), bins=80, alpha=0.5, color='steelblue', label='P3', density=True)
    ax.hist(rays[n_p3 : n_p3 + n_p4].flatten(), bins=80, alpha=0.5, color='darkorange', label='P4', density=True)
    ax.hist(rays[n_p3 + n_p4 :].flatten(), bins=80, alpha=0.5, color='seagreen', label='P5', density=True)
    ax.set_xlabel('Ray length (px)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    ax.set_title('Polygon Area Distribution per Scale', fontsize=10)
    # Approximate area via shoelace formula
    areas = np.zeros(len(rays))
    for i in range(len(rays)):
        v = vertices[i]
        x_coords = v[:, 0]
        y_coords = v[:, 1]
        areas[i] = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1)))

    ax.hist(areas[:n_p3], bins=60, alpha=0.5, color='steelblue', label='P3', density=True)
    ax.hist(areas[n_p3 : n_p3 + n_p4], bins=60, alpha=0.5, color='darkorange', label='P4', density=True)
    ax.hist(areas[n_p3 + n_p4 :], bins=60, alpha=0.5, color='seagreen', label='P5', density=True)
    ax.set_xlabel('Area (px²)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    ax.set_title('Summary Statistics', fontsize=10)
    stats_text = (
        f'Spatial Coverage:\n'
        f'  XY range X: [{xy[:, 0].min():.1f}, {xy[:, 0].max():.1f}]\n'
        f'  XY range Y: [{xy[:, 1].min():.1f}, {xy[:, 1].max():.1f}]\n'
        f'  Image size: {IMG_SIZE}x{IMG_SIZE}\n\n'
        f'Ray Statistics:\n'
        f'  P3 mean={rays[:n_p3].mean():.1f} std={rays[:n_p3].std():.1f}\n'
        f'  P4 mean={rays[n_p3 : n_p3 + n_p4].mean():.1f} std={rays[n_p3 : n_p3 + n_p4].std():.1f}\n'
        f'  P5 mean={rays[n_p3 + n_p4 :].mean():.1f} std={rays[n_p3 + n_p4 :].std():.1f}\n'
        f'  All min={rays.min():.2f} (>0 ✓)'
        if rays.min() > 0
        else f'  All min={rays.min():.2f} (ISSUE: ≤0!)'
    )
    stats_text += (
        f'\n\nArea Statistics:\n'
        f'  P3 mean={areas[:n_p3].mean():.0f} px²\n'
        f'  P4 mean={areas[n_p3 : n_p3 + n_p4].mean():.0f} px²\n'
        f'  P5 mean={areas[n_p3 + n_p4 :].mean():.0f} px²\n\n'
        f'Out-of-bounds check:\n'
        f'  Vertices outside [0, {IMG_SIZE}]:\n'
    )
    oob_x = ((vertices[:, :, 0] < 0) | (vertices[:, :, 0] > IMG_SIZE)).sum()
    oob_y = ((vertices[:, :, 1] < 0) | (vertices[:, :, 1] > IMG_SIZE)).sum()
    total_verts = vertices.shape[0] * vertices.shape[1]
    stats_text += f'  X: {oob_x}/{total_verts} ({oob_x / total_verts * 100:.1f}%)\n'
    stats_text += f'  Y: {oob_y}/{total_verts} ({oob_y / total_verts * 100:.1f}%)'

    ax.text(0.05, 0.92, stats_text, fontsize=9, transform=ax.transAxes, va='top', family='monospace')

    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / 'polygon_spread.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[INFO] saved -> {out}')


if __name__ == '__main__':
    generate_polygon_spread_plots()
