"""Audit held-out DETR misses from raw query predictions and dense PanNuke masks.

Use the cache produced by ``eval_lsp.py --save-predictions``. Matching follows
the evaluator's official overlap rule. Query geometry is inspected before
overlap resolution to separate missing geometry from background suppression.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from raycasted.evals.eval_lsp import ParquetGT, _poly_xy, rasterize_polys, roi_id_from_path


def _raw_mask(polygon: np.ndarray, size: int) -> np.ndarray:
    """Rasterize one polygon with the same PIL convention as LSP-DETR."""
    image = Image.new('1', (size, size))
    ImageDraw.Draw(image).polygon(_poly_xy(polygon).flatten().tolist(), outline=1, fill=1)
    return np.asarray(image, dtype=bool)


def _iou_matrix(preds: list[np.ndarray], targets: list[np.ndarray]) -> np.ndarray:
    """Return target-by-prediction mask IoU."""
    if not preds or not targets:
        return np.zeros((len(targets), len(preds)), dtype=np.float32)
    p = np.stack(preds).reshape(len(preds), -1).astype(np.float32)
    t = np.stack(targets).reshape(len(targets), -1).astype(np.float32)
    intersection = t @ p.T
    union = t.sum(1)[:, None] + p.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _candidate_query_ids(query_bounds: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Prune queries whose polygon bounding box cannot intersect a GT mask."""
    ys, xs = np.nonzero(target)
    if not len(xs):
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(
        (query_bounds[:, 1] + 1 >= xs.min())
        & (query_bounds[:, 0] - 1 <= xs.max())
        & (query_bounds[:, 3] + 1 >= ys.min())
        & (query_bounds[:, 2] - 1 <= ys.max())
    )


def audit(cache_path: Path, parquet_path: Path, *, max_images: int | None = None) -> tuple[dict, list[dict]]:
    """Summarize matching, type errors, duplicates, and missed-GT query geometry."""
    cache = np.load(cache_path)
    required = {'paths', 'offsets', 'polygons', 'labels', 'query_offsets', 'query_polygons', 'query_logits'}
    if not required.issubset(cache.files):
        raise ValueError(f'Cache lacks fields: {sorted(required - set(cache.files))}')
    paths = cache['paths']
    offsets = cache['offsets']
    polygons = cache['polygons']
    labels = cache['labels']
    query_offsets = cache['query_offsets']
    query_polygons = cache['query_polygons']
    query_logits_all = cache['query_logits']
    gt = ParquetGT(parquet_path)
    n_images = min(len(paths), max_images or len(paths))
    counts = Counter()
    rows = []
    for image_idx in range(n_images):
        roi = roi_id_from_path(str(paths[image_idx]))
        gt_masks, gt_labels, tissue = gt.get(roi)
        lo, hi = offsets[image_idx : image_idx + 2]
        qlo, qhi = query_offsets[image_idx : image_idx + 2]
        pred_polys = polygons[lo:hi]
        pred_labels = labels[lo:hi]
        query_polys = query_polygons[qlo:qhi]
        query_logits = query_logits_all[qlo:qhi]
        query_xy = np.stack([_poly_xy(poly) for poly in query_polys])
        query_bounds = np.stack(
            [
                query_xy[..., 0].min(1),
                query_xy[..., 0].max(1),
                query_xy[..., 1].min(1),
                query_xy[..., 1].max(1),
            ],
            axis=1,
        )
        size = gt_masks[0].shape[0] if gt_masks else 256
        pred_masks, resolved_labels = rasterize_polys(pred_polys, size, size, pred_labels)
        iou = _iou_matrix(pred_masks, gt_masks)
        match = {}
        if iou.size:
            rr, cc = linear_sum_assignment(-iou)
            match = {int(g): int(p) for g, p in zip(rr, cc) if iou[g, p] > 0.5}
        counts['images'] += 1
        counts['gt'] += len(gt_masks)
        counts['raw_predictions'] += len(pred_polys)
        counts['resolved_predictions'] += len(pred_masks)
        counts['wrong_type'] += sum(resolved_labels[p] != gt_labels[g] for g, p in match.items())
        counts['unmatched_predictions'] += len(pred_masks) - len(match)

        raw_pred_masks = [_raw_mask(poly, size) for poly in pred_polys]
        raw_iou = _iou_matrix(raw_pred_masks, gt_masks)
        for g in match:
            duplicates = int((raw_iou[g] > 0.5).sum()) - 1
            counts['duplicate_like_raw_predictions'] += max(duplicates, 0)

        centers = np.asarray([np.array(np.nonzero(mask)).mean(axis=1) for mask in gt_masks])
        for g, target in enumerate(gt_masks):
            area = int(target.sum())
            crowded = bool(
                len(centers) > 1
                and np.any((np.linalg.norm(centers - centers[g], axis=1) < 24) & (np.arange(len(centers)) != g))
            )
            if g in match:
                cohort = 'matched'
                best_iou = float(iou[g, match[g]])
                best_margin = None
                type_correct = bool(resolved_labels[match[g]] == gt_labels[g])
            else:
                type_correct = None
                candidates = _candidate_query_ids(query_bounds, target)
                target_area = area
                good = []
                best_iou = 0.0
                for q in candidates:
                    mask = _raw_mask(query_polys[q], size)
                    intersection = int((mask & target).sum())
                    if not intersection:
                        continue
                    value = intersection / (int(mask.sum()) + target_area - intersection)
                    best_iou = max(best_iou, value)
                    if value > 0.5:
                        good.append(int(q))
                if not good:
                    cohort = 'no_good_polygon'
                    best_margin = None
                else:
                    logits = query_logits[good]
                    margins = logits[:, :-1].max(axis=1) - logits[:, -1]
                    best_margin = float(margins.max())
                    cohort = 'good_suppressed_bg' if np.all(margins < 0) else 'good_fg_unmatched'
            counts[cohort] += 1
            rows.append(
                {
                    'roi': roi,
                    'gt_index': g,
                    'tissue': tissue,
                    'class_id': int(gt_labels[g]),
                    'area_px': area,
                    'crowded_24px': int(crowded),
                    'cohort': cohort,
                    'matched_type_correct': type_correct,
                    'best_query_iou': round(best_iou, 6),
                    'best_fg_minus_bg_logit': None if best_margin is None else round(best_margin, 6),
                }
            )
    summary = {
        'cache': str(cache_path),
        'images': n_images,
        'counts': {key: int(value) for key, value in counts.items()},
    }
    return summary, rows


def main() -> None:
    """Run a read-only cohort audit over a cached held-out prediction set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--max-images', type=int, default=384)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    summary, rows = audit(args.cache, args.parquet, max_images=args.max_images)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    with args.output_prefix.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
