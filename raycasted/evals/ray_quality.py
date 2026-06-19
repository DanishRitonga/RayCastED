"""Ray-count quality analysis: polygon IoU between GT mask and raycast reconstruction.

Loads PanNuke images, ingests with 8/16/32/64 rays, computes per-nucleus polygon IoU.
Saves visualizations as PNG and results as CSV.
"""

import csv
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from raycasted.data.etl.ops.convert import polygon_to_raycast
from raycasted.data.etl.utils import constants as _const
from shapely.geometry import Polygon as SPolygon

RAY_COUNTS = [8, 16, 32, 64]
OUT_DIR = Path('output/ray_quality')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pannuke_fold(fold=3):
    df = pd.read_parquet(f'raycasted/data/dataset/PanNuke/data/fold{fold}-00000-of-00001.parquet')
    return df


def extract_contours(mask):
    if mask.ndim == 3:
        mask = mask.max(axis=0) if mask.max() > 0 else mask[:, :, 0]
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def contour_to_mask(contour, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(m, [contour], -1, 1, -1)
    return m


def mask_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return inter / union if union > 0 else 0.0


def process_image(row, n_rays_list, save_pngs=False, img_idx=0):
    instances = row['instances']
    img_data = row['image']
    img = np.array(Image.open(io.BytesIO(img_data['bytes'])))
    h, w = img.shape[:2] if img.ndim >= 2 else (256, 256)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    gt_masks = []
    gt_contours = []
    for inst in instances:
        png = inst['bytes']
        mask = np.array(Image.open(io.BytesIO(png)))
        gt_masks.append(mask)
        contours = extract_contours(mask)
        if contours:
            gt_contours.append(max(contours, key=cv2.contourArea))

    results = {}
    for n_rays in n_rays_list:
        _const.configure_rays(n_rays)
        ious = []
        polygons = []
        for contour in gt_contours:
            verts = contour.squeeze(1).astype(np.float64)
            if verts.ndim < 2 or len(verts) < 3:
                continue
            ann = polygon_to_raycast(SPolygon(verts), 0, n_rays=n_rays)
            if ann is None:
                continue
            cx, cy = ann[1], ann[2]
            rays = ann[3 : 3 + n_rays]
            cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
            sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
            vx = cx + rays * cos
            vy = cy + rays * sin
            pts = np.stack([vx, vy], axis=-1).clip(-32768, 32767).astype(np.int32).reshape(-1, 1, 2)
            ray_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(ray_mask, [pts], 1)
            gt_mask = contour_to_mask(contour, h, w)
            iou = mask_iou(ray_mask, gt_mask)
            ious.append(iou)
            polygons.append({'cx': cx, 'cy': cy, 'rays': rays})
        results[n_rays] = {'ious': ious, 'polygons': polygons, 'n': len(gt_contours)}

    if save_pngs:
        for n_rays in n_rays_list:
            vis = img.copy()
            for poly in results[n_rays]['polygons']:
                cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
                sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
                vx = poly['cx'] + poly['rays'] * cos
                vy = poly['cy'] + poly['rays'] * sin
                pts = np.stack([vx, vy], axis=-1).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
                cv2.circle(vis, (int(poly['cx']), int(poly['cy'])), 3, (0, 0, 255), -1)
            cv2.imwrite(str(OUT_DIR / f'img{img_idx:04d}_ray{n_rays}.png'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        gt_vis = img.copy()
        for c in gt_contours:
            cv2.drawContours(gt_vis, [c], -1, (0, 255, 0), 2)
        cv2.imwrite(str(OUT_DIR / f'img{img_idx:04d}_gt.png'), cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))

    return results


def main():
    import argparse, sys

    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=3)
    p.add_argument('--max-images', type=int, default=0, help='0=all')
    p.add_argument('--save-first', type=int, default=5, help='Save PNGs for first N images')
    args = p.parse_args()

    df = load_pannuke_fold(fold=args.fold)
    n_total = min(args.max_images, len(df)) if args.max_images > 0 else len(df)
    print(f'Fold {args.fold}: {len(df)} images, processing {n_total}')

    agg = {n: [] for n in RAY_COUNTS}
    for i in range(n_total):
        row = df.iloc[i]
        save = i < args.save_first
        r = process_image(row, RAY_COUNTS, save_pngs=save, img_idx=i)
        for n_rays in RAY_COUNTS:
            agg[n_rays].extend(r[n_rays]['ious'])
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{n_total}', flush=True)

    print(f'\n{"=" * 60}')
    print(f'  Ray-count quality (Fold {args.fold}, {n_total} images, {len(agg[8])} nuclei)')
    print(f'{"=" * 60}')
    for n_rays in RAY_COUNTS:
        a = np.array(agg[n_rays])
        print(
            f'  {n_rays:>3} rays: mean={a.mean():.4f} ± {a.std():.4f}, min={a.min():.4f}, median={np.median(a):.4f}, max={a.max():.4f}'
        )
    print(f'{"=" * 60}')

    # Save CSV
    csv_path = OUT_DIR / f'ray_quality_fold{args.fold}_n{n_total}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_rays', 'nuclei', 'mean', 'std', 'median', 'min', 'max'])
        for n_rays in RAY_COUNTS:
            a = np.array(agg[n_rays])
            w.writerow(
                [
                    n_rays,
                    len(a),
                    f'{a.mean():.4f}',
                    f'{a.std():.4f}',
                    f'{np.median(a):.4f}',
                    f'{a.min():.4f}',
                    f'{a.max():.4f}',
                ]
            )
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()
