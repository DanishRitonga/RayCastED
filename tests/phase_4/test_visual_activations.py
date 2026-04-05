"""Phase 4 visual — Activation sanity plots.

Histograms of raw vs activated values for xy and ray channels
across the full anchor grid (8400 anchors, 3 scales).

Run with: uv run python tests/phase_4/test_visual_activations.py
"""

import torch
import torch.nn.functional as F

from raycasted.model.head import RAYCAST_DIM, RayCastDetect

NC = 4
REG_MAX = 16
CH = (64, 128, 256)
FEAT_SIZE = 80


def _make_feats(batch=2, feat_size=FEAT_SIZE, ch=CH):
    return [torch.randn(batch, c, feat_size // (2**i), feat_size // (2**i)) for i, c in enumerate(ch)]


def generate_activation_plots():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    head = RayCastDetect(nc=NC, reg_max=REG_MAX, end2end=False, ch=CH)
    head.eval()
    head.stride = torch.tensor([8.0, 16.0, 32.0])

    feats = _make_feats(batch=1)
    with torch.no_grad():
        preds = head.forward_head(feats, box_head=head.cv2, cls_head=head.cv3)

    poly = preds['boxes']  # [1, 34, 8400]
    xy_raw = poly[0, :2, :].flatten()  # all xy logits
    ray_raw = poly[0, 2:, :].flatten()  # all ray logits
    xy_act = xy_raw.sigmoid()
    ray_act = F.softplus(ray_raw)

    # Per-scale split
    n_p3 = 80 * 80  # 6400
    n_p4 = 40 * 40  # 1600
    n_p5 = 20 * 20  # 400

    ray_p3 = poly[0, 2:, :n_p3]
    ray_p4 = poly[0, 2:, n_p3 : n_p3 + n_p4]
    ray_p5 = poly[0, 2:, n_p3 + n_p4 :]

    ray_p3_act = F.softplus(ray_p3)
    ray_p4_act = F.softplus(ray_p4)
    ray_p5_act = F.softplus(ray_p5)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase 4 — Activation Sanity Plots', fontsize=14, fontweight='bold')

    # --- Row 1: XY channel ---
    ax = axes[0, 0]
    ax.hist(xy_raw.numpy(), bins=100, color='steelblue', alpha=0.8, edgecolor='none')
    ax.set_title('XY Raw Logits', fontsize=11)
    ax.set_xlabel('Logit value')
    ax.set_ylabel('Count')
    ax.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)

    ax = axes[0, 1]
    ax.hist(xy_act.numpy(), bins=100, color='darkorange', alpha=0.8, edgecolor='none')
    ax.set_title('XY After Sigmoid', fontsize=11)
    ax.set_xlabel('Sigmoid value')
    ax.set_ylabel('Count')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 1)
    # Mark saturation zones
    ax.axvspan(0, 0.05, color='red', alpha=0.1)
    ax.axvspan(0.95, 1.0, color='red', alpha=0.1)

    ax = axes[0, 2]
    ax.text(
        0.05,
        0.95,
        'XY Statistics',
        fontsize=12,
        fontweight='bold',
        transform=ax.transAxes,
        va='top',
        family='monospace',
    )
    stats_text = (
        f'Raw logits:\n'
        f'  mean={xy_raw.mean():.3f}  std={xy_raw.std():.3f}\n'
        f'  min ={xy_raw.min():.3f}  max={xy_raw.max():.3f}\n\n'
        f'After Sigmoid:\n'
        f'  mean={xy_act.mean():.3f}  std={xy_act.std():.3f}\n'
        f'  min ={xy_act.min():.3f}  max={xy_act.max():.3f}\n\n'
        f'Saturation (< 0.05 or > 0.95):\n'
        f'  {((xy_act < 0.05) | (xy_act > 0.95)).float().mean() * 100:.1f}% of values\n\n'
        f'Near-centre (0.3 - 0.7):\n'
        f'  {((xy_act > 0.3) & (xy_act < 0.7)).float().mean() * 100:.1f}% of values'
    )
    ax.text(0.05, 0.85, stats_text, fontsize=9, transform=ax.transAxes, va='top', family='monospace')
    ax.axis('off')

    # --- Row 2: Ray channels ---
    ax = axes[1, 0]
    ax.hist(ray_raw.numpy(), bins=100, color='steelblue', alpha=0.8, edgecolor='none')
    ax.set_title('Ray Raw Logits', fontsize=11)
    ax.set_xlabel('Logit value')
    ax.set_ylabel('Count')
    ax.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)

    ax = axes[1, 1]
    ax.hist(ray_act.numpy(), bins=100, color='seagreen', alpha=0.8, edgecolor='none')
    ax.set_title('Ray After Softplus', fontsize=11)
    ax.set_xlabel('Softplus value')
    ax.set_ylabel('Count')
    # Mark minimum (ln(2) ≈ 0.693)
    ax.axvline(
        torch.log(torch.tensor(2.0)).item(), color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='min=ln(2)'
    )
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.text(
        0.05,
        0.95,
        'Ray Statistics',
        fontsize=12,
        fontweight='bold',
        transform=ax.transAxes,
        va='top',
        family='monospace',
    )
    stats_text = (
        f'Raw logits:\n'
        f'  mean={ray_raw.mean():.3f}  std={ray_raw.std():.3f}\n'
        f'  min ={ray_raw.min():.3f}  max={ray_raw.max():.3f}\n\n'
        f'After Softplus:\n'
        f'  mean={ray_act.mean():.3f}  std={ray_act.std():.3f}\n'
        f'  min ={ray_act.min():.4f}  max={ray_act.max():.3f}\n'
        f'  (min must be > 0)\n\n'
        f'Per-scale ray mean (grid space):\n'
        f'  P3 (s=8):  {ray_p3_act.mean():.3f}\n'
        f'  P4 (s=16): {ray_p4_act.mean():.3f}\n'
        f'  P5 (s=32): {ray_p5_act.mean():.3f}\n\n'
        f'Near-zero (< 0.5):\n'
        f'  {(ray_act < 0.5).float().mean() * 100:.1f}% of values'
    )
    ax.text(0.05, 0.85, stats_text, fontsize=9, transform=ax.transAxes, va='top', family='monospace')
    ax.axis('off')

    plt.tight_layout()
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / 'activation_sanity.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[INFO] saved -> {out}')


if __name__ == '__main__':
    generate_activation_activations = generate_activation_plots()
