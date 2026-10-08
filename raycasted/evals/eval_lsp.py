r"""PanNuke evaluation matching the LSP-DETR protocol EXACTLY.

Metrics replicated from the LSP-DETR codebase
(https://github.com/RationAI/lsp-detr, metrics/binary_panoptic_quality.py +
metrics/panoptic_quality.py + metrics/f1_score.py + det_meta_arch.py):

  - bPQ / bDQ / bSQ : PER-IMAGE binary PQ, then MEAN OVER IMAGES.
        Per image: IoU matrix (preds x targets), linear_sum_assignment on
        -IoU, keep matched IoU > 0.5, tp = #kept,
        dq = tp / (0.5*P + 0.5*M + eps), sq = sum(matched iou) / tp,
        pq = dq * sq. Images with empty targets are skipped (no count).
  - mPQ / mDQ / mSQ : PER-IMAGE mean over GT-present classes (for each class
        c present in the GT of that image, binary PQ of preds[cls==c] vs
        targets[cls==c]), then MEAN OVER IMAGES.
  - bMPQ/mMPQ masked variants (masked=True): pred pixels overlapping ANY GT
        are zeroed before counting (masked_preds = preds & ~union(gt)).
  - F1 : centroid distance Hungarian, radius 12 px, GLOBAL tp/fp/fn sums.
  - Labels: logits.argmax over nc+1, dropping the no-object (last) column.
        NO confidence threshold is applied at evaluation. Duplicate-trained
        checkpoints additionally use class-agnostic polygon NMS by default;
        pass --raw-duplicates for the unsuppressed diagnostic.
  - 'Average' row = macro mean over per-tissue means; STD across the 3 CV
        folds (computed across runs by the caller, not here).

Usage:
    uv run python -m raycasted.evals.eval_lsp \\
        --weights output/f1-train/train/weights/best.pt \\
        --data-dir output/f1-train/transformed/test \\
        --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \\
        --batch 16 --device 0
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.evals.eval_pannuke import load_model
from raycasted.evals.polygon_nms import polygon_nms_indices
from raycasted.model.register import register_raycast_head

TISSUE_NAMES = [
    'Adrenal Gland',
    'Bile Duct',
    'Bladder',
    'Breast',
    'Cervix',
    'Colon',
    'Esophagus',
    'Head&Neck',
    'Kidney',
    'Liver',
    'Lung',
    'Ovarian',
    'Pancreatic',
    'Prostate',
    'Skin',
    'Stomach',
    'Testis',
    'Thyroid',
    'Uterus',
]


# ---------------------------------------------------------------------------
# Per-image PQ helpers (LSP-DETR exact)
# ---------------------------------------------------------------------------


def _binary_pq(pred_masks, gt_masks, threshold=0.5):
    """Per-image binary PQ (LSP BinaryPanopticQuality.update).

    pred_masks/gt_masks: lists of [H, W] bool masks. Empty targets -> None.
    Returns (dq, sq, pq) or None when targets empty.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_gt == 0:
        return None
    if n_pred == 0:
        # LSP: intersection is (0, n_gt); assignment empty -> tp 0.
        return 0.0, 0.0, 0.0

    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)

    intersection = pred_stack @ gt_stack.T  # (n_pred, n_gt)
    pred_area = pred_stack.sum(1, keepdims=True)
    gt_area = gt_stack.sum(1, keepdims=True)
    union = pred_area + gt_area.T - intersection
    union = np.where(union <= 0, 1.0, union)
    iou = intersection / union

    row, col = linear_sum_assignment(-iou)
    matched = iou[row, col]
    valid = matched > threshold
    tp = int(valid.sum())
    eps = 1e-6
    dq = tp / (0.5 * n_pred + 0.5 * n_gt + eps)
    sq = matched[valid].sum() / (tp + eps)
    pq = dq * sq
    return dq, sq, pq


def _binary_pq_masked(pred_masks, gt_masks, threshold=0.5):
    """LSP masked=True variant: pred pixels inside ANY GT are zeroed.

    union = gt_area + masked_pred_area  (no -intersection, since masked pred
    pixels are disjoint from GT by construction). intersection unchanged.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_gt == 0:
        return None
    if n_pred == 0:
        return 0.0, 0.0, 0.0

    gt_union = np.stack(gt_masks).any(0) if n_gt > 0 else np.zeros(pred_masks[0].shape, dtype=bool)
    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)

    intersection = pred_stack @ gt_stack.T
    masked_pred_area = pred_stack[:, gt_union.reshape(-1) == 0].sum(1) if gt_union.any() else pred_stack.sum(1)
    # masked_pred_area[i] = pixels of pred i outside any GT
    gt_area = gt_stack.sum(1, keepdims=True)  # (n_gt, 1)
    union = gt_area.T + masked_pred_area[:, None]  # (n_pred, n_gt)
    union = np.where(union <= 0, 1.0, union)
    iou = intersection / union

    row, col = linear_sum_assignment(-iou)
    matched = iou[row, col]
    valid = matched > threshold
    tp = int(valid.sum())
    eps = 1e-6
    dq = tp / (0.5 * n_pred + 0.5 * n_gt + eps)
    sq = matched[valid].sum() / (tp + eps)
    pq = dq * sq
    return dq, sq, pq


# ---------------------------------------------------------------------------
# Dense GT loading from fold parquet
# ---------------------------------------------------------------------------


def _decode_instance_mask(bytes_):
    """Decode a stored instance PNG into a (256, 256) bool mask."""
    arr = cv2.imdecode(np.frombuffer(bytes_, np.uint8), cv2.IMREAD_UNCHANGED)
    return arr > 127


class ParquetGT:
    """Hold per-ROI dense instance masks + labels + tissue for one fold.

    Instance masks are decoded lazily in ``get`` (one ROI at a time) so a
    whole-fold run never materializes every GT mask simultaneously (~4.4 GB
    for 66k instances at 256x256 bool each).
    """

    def __init__(self, parquet_path):
        self.df = pl.read_parquet(parquet_path)
        self.rows = []  # indexed by roi_id (== row order)
        for row in self.df.iter_rows(named=True):
            insts = row.get('instances') or []
            cats = row.get('categories') or []
            self.rows.append(([i['bytes'] for i in insts], [int(c) for c in cats], int(row['tissue'])))
        self.df = None  # drop the polars frame once rows are extracted

    def __len__(self):
        """Number of ROIs in the fold."""
        return len(self.rows)

    def get(self, roi_id):
        """Return (masks, labels, tissue) for the given roi id (decoded now)."""
        inst_bytes, cats, tissue = self.rows[roi_id]
        masks = []
        labels = []
        for blob, cat in zip(inst_bytes, cats):
            m = _decode_instance_mask(blob)
            if m.any():
                masks.append(m)
                labels.append(cat)
        return masks, np.asarray(labels, dtype=int), tissue


def roi_id_from_path(path: str) -> int:
    """Extract the numeric ROI id from a tile filename."""
    m = re.search(r'_roi_(\d+)', str(path))
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Inference: decode raw DETR outputs the LSP way (no conf threshold)
# ---------------------------------------------------------------------------


def run_inference_raw(model, dataloader, device):
    """Run model and emit per-image LSP-style decoded detections.

    Each element: dict(polys=(N, 66) pixel coords, labels=(N,) int in [0, nc),
    scores=(N,) foreground confidence)
    from logits.argmax over nc+1 with the no-object (last) column dropped.
    """
    results = []
    tr = getattr(model, 'training_args', {})
    crop = int(tr.get('crop_size', 256))
    no_object = bool(getattr(model, 'no_object', False))
    nc = int(tr.get('nc', getattr(model, 'nc', 5)))
    _ = nc
    n_rays = int(tr.get('n_rays', 64))
    _ = n_rays

    with torch.no_grad():
        for batch in dataloader:
            images = batch['img'].to(device)
            raw_out = model(images)
            # model(x) eval -> (y_detect, out5) where out5 = decoder's
            # (dec_polygons, dec_scores, enc_polygons, enc_scores, dn_meta).
            # dec_polygons (1,B,nq,66) NORMALIZED, dec_scores (1,B,nq,nc+1) logits.
            y_detect, out5 = raw_out if isinstance(raw_out, tuple) else (raw_out, None)
            dec_polys = out5[0]  # (1, B, nq, raycast_dim)
            dec_logits = out5[1]  # (1, B, nq, nc+1)
            dec_polys = dec_polys.squeeze(0)  # (B, nq, 66)
            dec_logits = dec_logits.squeeze(0)  # (B, nq, nc+1)
            bs = images.shape[0]
            for si in range(bs):
                poly_n = dec_polys[si].cpu().numpy()  # normalized [0,1]
                logits = dec_logits[si].cpu().numpy()
                if no_object:
                    keep = logits.argmax(-1) != (logits.shape[-1] - 1)
                    scores = 1.0 - torch.sigmoid(torch.from_numpy(logits[:, -1])).numpy()
                else:
                    keep = np.ones(logits.shape[0], dtype=bool)
                    scores = torch.sigmoid(torch.from_numpy(logits)).max(dim=-1).values.numpy()
                poly_n = poly_n[keep]
                labels = logits[keep].argmax(-1)
                scores = scores[keep]
                # normalized -> pixel (square crop == imgsz)
                poly_px = poly_n * crop
                results.append({'polys': poly_px, 'labels': labels.astype(int), 'scores': scores})
    return results


# ---------------------------------------------------------------------------
# Metric accumulation (per-tissue)
# ---------------------------------------------------------------------------


class TissueStats:
    """Accumulate per-tissue/per-image PQ + centroid F1 statistics."""

    def __init__(self):
        """Initialize all metric accumulators to zero."""
        self.bpq_s = self.bdq_s = self.bsq_s = 0.0
        self.bmpq_s = self.bmdq_s = self.bmsq_s = 0.0
        self.mpq_s = self.mdq_s = self.msq_s = 0.0
        self.mmpq_s = self.mmdq_s = self.mmsq_s = 0.0
        self.b_count = 0
        self.m_count = 0
        # centroid F1 global sums
        self.f1_tp = self.f1_fp = self.f1_fn = 0

    def update(self, pred_masks, pred_labels, gt_masks, gt_labels):
        """Accumulate PQ (+ masked + multiclass) for one image."""
        r = _binary_pq(pred_masks, gt_masks)
        if r is not None:
            dq, sq, pq = r
            self.bdq_s += dq
            self.bsq_s += sq
            self.bpq_s += pq
            self.b_count += 1
        rm = _binary_pq_masked(pred_masks, gt_masks)
        if rm is not None:
            dq, sq, pq = rm
            self.bmdq_s += dq
            self.bmsq_s += sq
            self.bmpq_s += pq

        # multiclass PQ over GT-present classes
        present = set(int(c) for c in np.unique(gt_labels))
        if present:
            m_dqs, m_sqs, m_pqs = [], [], []
            mm_dqs, mm_sqs, mm_pqs = [], [], []
            for c in present:
                p = [m for m, lbl in zip(pred_masks, pred_labels) if lbl == c]
                g = [m for m, lbl in zip(gt_masks, gt_labels) if lbl == c]
                rc = _binary_pq(p, g)
                if rc is not None:
                    m_dqs.append(rc[0])
                    m_sqs.append(rc[1])
                    m_pqs.append(rc[2])
                rmc = _binary_pq_masked(p, g)
                if rmc is not None:
                    mm_dqs.append(rmc[0])
                    mm_sqs.append(rmc[1])
                    mm_pqs.append(rmc[2])
            if m_dqs:
                self.mdq_s += float(np.mean(m_dqs))
                self.msq_s += float(np.mean(m_sqs))
                self.mpq_s += float(np.mean(m_pqs))
                self.m_count += 1
            if mm_dqs:
                self.mmdq_s += float(np.mean(mm_dqs))
                self.mmsq_s += float(np.mean(mm_sqs))
                self.mmpq_s += float(np.mean(mm_pqs))

    def update_f1(self, pred_masks, gt_masks, radius=12):
        """Accumulate global centroid-based F1 tp/fp/fn for one image."""

        # centroid-based F1 (global tp/fp/fn). GT centroids vs pred centroids.
        def _centroids(masks):
            pts = []
            for m in masks:
                ys, xs = np.nonzero(m)
                if len(ys) == 0:
                    continue
                pts.append((xs.mean(), ys.mean()))
            return np.asarray(pts).reshape(-1, 2) if pts else np.zeros((0, 2))

        pc = _centroids(pred_masks)
        gc = _centroids(gt_masks)
        tp = fp = fn = 0
        if len(gc) == 0:
            fn = 0  # no GT -> no FN by LSP (mask empty)
            fp = len(pc)
        elif len(pc) == 0:
            fp = 0
            fn = len(gc)
        else:
            dist = np.linalg.norm(pc[:, None, :] - gc[None, :, :], axis=-1)  # (P, M)
            row, col = linear_sum_assignment(dist)
            valid = dist[row, col] <= radius
            tp = int(valid.sum())
            fp = len(pc) - tp
            fn = len(gc) - tp
        self.f1_tp += tp
        self.f1_fp += fp
        self.f1_fn += fn

    def f1(self):
        """Return (F1, precision, recall) from accumulated global counts."""
        eps = 1e-6
        tp, fp, fn = self.f1_tp, self.f1_fp, self.f1_fn
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        return 2 * p * r / (p + r + eps), p, r

    def summary(self):
        """Return dict of per-image-mean PQ metrics over accumulated images."""

        def _avg(s, n):
            return s / n if n else float('nan')

        return {
            'bDQ': _avg(self.bdq_s, self.b_count),
            'bSQ': _avg(self.bsq_s, self.b_count),
            'bPQ': _avg(self.bpq_s, self.b_count),
            'bMDQ': _avg(self.bmdq_s, self.b_count),
            'bMSQ': _avg(self.bmsq_s, self.b_count),
            'bMPQ': _avg(self.bmpq_s, self.b_count),
            'mDQ': _avg(self.mdq_s, self.m_count),
            'mSQ': _avg(self.msq_s, self.m_count),
            'mPQ': _avg(self.mpq_s, self.m_count),
            'mMDQ': _avg(self.mmdq_s, self.m_count),
            'mMSQ': _avg(self.mmsq_s, self.m_count),
            'mMPQ': _avg(self.mmpq_s, self.m_count),
        }


def rasterize_polys(polys, h, w, labels):
    """Rasterize (N, 66) pixel polys, largest-area-first occlusion."""
    masks = []
    kept = []
    order = np.argsort([-_poly_area(p) for p in polys]) if len(polys) else []
    occupied = np.zeros((h, w), dtype=bool)
    for i in order:
        m = np.zeros((h, w), dtype=np.uint8)
        pts = _poly_xy(polys[i])
        cv2.fillPoly(m, [pts], 1)
        m = m.astype(bool) & ~occupied
        if m.any():
            masks.append(m)
            kept.append(labels[i])
            occupied |= m
    return masks, np.asarray(kept, dtype=int) if kept else np.zeros((0,), dtype=int)


def rasterize_polys_watershed(polys, h, w, labels):
    """LSP distance-transform watershed overlap resolution over pred polys.

    Each polygon is filled independently (no occlusion merge), then every
    pixel is assigned to the instance whose EDT of its own mask is largest
    there (nearest center). Overlapping predictions can therefore BOTH
    survive, each keeping the pixels closest to its center. Mirrors LSP-DETR
    image_processing.py resolve_nuclei_overlaps used at PanNuke eval/val.
    """
    n = len(polys)
    if n == 0:
        return [], np.zeros((0,), dtype=int)

    inst = np.zeros((n, h, w), dtype=bool)
    for i, poly in enumerate(polys):
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [_poly_xy(poly)], 1)
        inst[i] = m.astype(bool)

    dist = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        if inst[i].any():
            dist[i] = distance_transform_edt(inst[i])

    owner = dist.argmax(0)  # per-pixel nearest center (torch.max first-max)
    owned = np.arange(n)[:, None, None] == owner[None]
    resolved = owned & (dist > 0)

    masks = []
    kept = []
    for i in range(n):
        m = resolved[i]
        if m.any():
            masks.append(m)
            kept.append(labels[i])
    return masks, np.asarray(kept, dtype=int) if kept else np.zeros((0,), dtype=int)


def _poly_area(poly):
    cx, cy = poly[0], poly[1]
    rays = poly[2:]
    n = len(rays)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = cx + rays * np.cos(angles)
    ys = cy + rays * np.sin(angles)
    return 0.5 * np.abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def _poly_xy(poly):
    cx, cy = poly[0], poly[1]
    rays = poly[2:]
    n = len(rays)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = cx + rays * np.cos(angles)
    ys = cy + rays * np.sin(angles)
    return np.stack([xs, ys], 1).astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def collate_fn(batch):
    """Stack only the image tensor from RayCastTileDataset 3-tuples."""
    import torch as _t

    images = _t.stack([item[0] for item in batch])
    return {'img': images}


def main():
    """Run LSP-protocol eval: dense GT vs decoded preds, per-tissue PQ."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data-dir', required=True, help='test tiles dir (.npz)')
    ap.add_argument('--parquet', required=True, help='fold parquet with dense GT')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument(
        '--polygon-nms-iou',
        type=float,
        default=None,
        help='class-agnostic polygon NMS IoU; auto-enables from checkpoint for duplicate-trained models',
    )
    ap.add_argument(
        '--raw-duplicates',
        action='store_true',
        help='disable checkpoint-inferred polygon NMS and evaluate raw duplicate predictions',
    )
    ap.add_argument(
        '--watershed',
        action='store_true',
        help='also resolve pred overlaps via LSP distance-transform '
        'watershed and print the delta vs largest-first occlusion',
    )
    args = ap.parse_args()

    register_raycast_head()
    from raycasted.data.etl.utils import constants as _const

    _const.configure_rays(64)

    device = torch.device('cuda' if args.device != 'cpu' else 'cpu')
    model = load_model(args.weights, device)
    tr = getattr(model, 'training_args', {})
    crop = int(tr.get('crop_size', 256))
    nms_iou = args.polygon_nms_iou
    if nms_iou is None and not args.raw_duplicates and int(tr.get('detr_duplicate_train_topk', 1)) > 1:
        nms_iou = float(tr.get('detr_duplicate_nms_iou', 0.5))
    if nms_iou is not None and not 0.0 <= nms_iou <= 1.0:
        raise ValueError('--polygon-nms-iou must be in [0, 1]')

    # map tile path -> roi id
    tile_paths = sorted(Path(args.data_dir).glob('*.npz'))
    gt = ParquetGT(args.parquet)
    assert len(tile_paths) == len(gt), (len(tile_paths), len(gt))

    ds = RayCastTileDataset(str(args.data_dir), crop_size=crop, augment=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    preds = run_inference_raw(model, dl, device)

    per_tissue = {}
    total = TissueStats()
    per_tissue_ws = {} if args.watershed else None
    total_ws = TissueStats() if args.watershed else None
    n_pred_total = n_pred_raw_total = n_gt_total = 0
    n_pred_total_ws = 0
    for path, res in zip(tile_paths, preds):
        roi = roi_id_from_path(path)
        gt_masks, gt_labels, tissue = gt.get(roi)
        pred_polys = res['polys']
        pred_labels = res['labels']
        pred_scores = res['scores']
        n_pred_raw_total += len(pred_polys)
        if nms_iou is not None:
            keep = polygon_nms_indices(
                pred_polys,
                pred_scores,
                iou_threshold=nms_iou,
                class_ids=pred_labels,
                class_agnostic=True,
            )
            pred_polys = pred_polys[keep]
            pred_labels = pred_labels[keep]
        if len(pred_polys) == 0:
            pm, pl = [], np.zeros((0,), dtype=int)
            ws_pm, ws_pl = [], np.zeros((0,), dtype=int)
        else:
            pm, pl = rasterize_polys(pred_polys, crop, crop, pred_labels)
            if args.watershed:
                ws_pm, ws_pl = rasterize_polys_watershed(pred_polys, crop, crop, pred_labels)
            else:
                ws_pm, ws_pl = [], np.zeros((0,), dtype=int)
        n_pred_total += len(pm)
        n_pred_total_ws += len(ws_pm)
        n_gt_total += len(gt_masks)
        ts = per_tissue.setdefault(tissue, TissueStats())
        ts.update(pm, pl, gt_masks, gt_labels)
        ts.update_f1(pm, gt_masks)
        total.update(pm, pl, gt_masks, gt_labels)
        total.update_f1(pm, gt_masks)
        if args.watershed:
            tsw = per_tissue_ws.setdefault(tissue, TissueStats())
            tsw.update(ws_pm, ws_pl, gt_masks, gt_labels)
            tsw.update_f1(ws_pm, gt_masks)
            total_ws.update(ws_pm, ws_pl, gt_masks, gt_labels)
            total_ws.update_f1(ws_pm, gt_masks)

    print(f'GT instances: {n_gt_total}   Raw pred instances (all non-∅): {n_pred_raw_total}')
    if nms_iou is not None:
        print(
            f'Class-agnostic polygon NMS IoU={nms_iou:g}: {n_pred_total} instances '
            f'(keep {n_pred_total / max(n_pred_raw_total, 1):.3f})'
        )
    else:
        print(f'Evaluated pred instances: {n_pred_total} (raw; polygon NMS disabled)')
    print(f'Per-image-mean bPQ (whole fold): {total.summary()["bPQ"]:.4f}')
    print(f'Per-image-mean mPQ (whole fold): {total.summary()["mPQ"]:.4f}')
    print()
    print(f'{"tissue":<15}{"bDQ":>8}{"bSQ":>8}{"bPQ":>8}{"mPQ":>8}{"F1":>8}{"P":>8}{"R":>8}')
    bpq_list, mpq_list = [], []
    for t in sorted(per_tissue):
        ts = per_tissue[t]
        s = ts.summary()
        f1, p, r = ts.f1()
        bpq_list.append(s['bPQ'])
        mpq_list.append(s['mPQ'])
        name = TISSUE_NAMES[t] if t < len(TISSUE_NAMES) else str(t)
        print(f'{name:<15}{s["bDQ"]:>8.4f}{s["bSQ"]:>8.4f}{s["bPQ"]:>8.4f}{s["mPQ"]:>8.4f}{f1:>8.4f}{p:>8.4f}{r:>8.4f}')
    if bpq_list:
        print(
            f'{"Average":<15}{"":>8}{"":>8}{np.mean(bpq_list):>8.4f}'
            f'{np.mean(mpq_list):>8.4f}   (macro over {len(bpq_list)} tissues)'
        )
    print()
    st = total.summary()
    f1, p, r = total.f1()
    print('Whole-fold totals:')
    for k in ['bDQ', 'bSQ', 'bPQ', 'bMPQ', 'mDQ', 'mSQ', 'mPQ', 'mMPQ']:
        print(f'  {k}: {st[k]:.4f}')
    print(f'  F1: {f1:.4f}  P: {p:.4f}  R: {r:.4f}  (centroid, r=12, global)')

    if args.watershed:
        print()
        print('=== LSP watershed overlap resolution (resolve_nuclei_overlaps) ===')
        print(f'Pred instances: occlusion {n_pred_total}   watershed {n_pred_total_ws}')
        print(f'{"tissue":<15}{"bDQ":>8}{"bSQ":>8}{"bPQ":>8}{"mPQ":>8}{"F1":>8}{"P":>8}{"R":>8}')
        ws_bpq, ws_mpq = [], []
        for t in sorted(per_tissue_ws):
            ts = per_tissue_ws[t]
            s = ts.summary()
            f1_w, p_w, r_w = ts.f1()
            ws_bpq.append(s['bPQ'])
            ws_mpq.append(s['mPQ'])
            name = TISSUE_NAMES[t] if t < len(TISSUE_NAMES) else str(t)
            print(
                f'{name:<15}{s["bDQ"]:>8.4f}{s["bSQ"]:>8.4f}{s["bPQ"]:>8.4f}'
                f'{s["mPQ"]:>8.4f}{f1_w:>8.4f}{p_w:>8.4f}{r_w:>8.4f}'
            )
        if ws_bpq:
            print(
                f'{"Average":<15}{"":>8}{"":>8}{np.mean(ws_bpq):>8.4f}'
                f'{np.mean(ws_mpq):>8.4f}   (macro over {len(ws_bpq)} tissues)'
            )
        st_ws = total_ws.summary()
        f1_ws, p_ws, r_ws = total_ws.f1()
        print()
        print('Whole-fold totals (watershed):')
        for k in ['bDQ', 'bSQ', 'bPQ', 'bMPQ', 'mDQ', 'mSQ', 'mPQ', 'mMPQ']:
            print(f'  {k}: {st_ws[k]:.4f}')
        print(f'  F1: {f1_ws:.4f}  P: {p_ws:.4f}  R: {r_ws:.4f}  (centroid, r=12, global)')
        print()
        print('delta (watershed - occlusion), per-image-mean whole fold:')
        for k in ['bPQ', 'mPQ', 'bDQ', 'bSQ', 'mDQ', 'mSQ']:
            if k in st_ws and k in st:
                d = st_ws[k] - st[k]
                print(f'  {k}: {d:+.4f}')
        d_f1 = f1_ws - f1
        d_p = p_ws - p
        d_r = r_ws - r
        print(f'  F1: {d_f1:+.4f}  P: {d_p:+.4f}  R: {d_r:+.4f}')


if __name__ == '__main__':
    main()
