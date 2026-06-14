"""Zero-Shot Generalization Eval — binary nuclei detection on external datasets.

Loads a PanNuke-trained model, runs inference on external test tiles,
maps all class predictions to binary (fg/bg), computes AJI, bPQ, F1, P, R.

Usage:
    uv run python main/eval_zero_shot.py \
        --weights output/v1.1/weights/best.pt \
        --data-dir output/puma/transformed/test \
        --dataset PUMA --conf 0.49
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.utils import constants as _const
from raycasted.model.metrics import compute_aji, resolve_mask_overlaps
from raycasted.model.register import register_raycast_head


def load_model(weights_path, device):
    register_raycast_head()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt.get('model') or ckpt.get('ema') if isinstance(ckpt, dict) else ckpt
    return model.float().to(device).eval()


def run_inference(model, dataloader, device, conf_threshold, raycast_dim):
    results = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['img'].to(device)
            raw_out = model(images)
            decoded = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            for si in range(images.shape[0]):
                det = decoded[si].cpu().numpy()
                if det.ndim == 2 and det.shape[1] >= raycast_dim + 2:
                    conf_mask = det[:, raycast_dim] > conf_threshold
                    det = det[conf_mask]
                else:
                    det = np.zeros((0, raycast_dim + 2), dtype=np.float32)

                mask = batch['batch_idx'] == si
                gt_poly = batch['bboxes'][mask].numpy()
                crop_size = int(dataloader.dataset.crop_size)
                if gt_poly.shape[0] > 0:
                    gt_poly = gt_poly.copy()
                    gt_poly[:, 0] *= crop_size
                    gt_poly[:, 1] *= crop_size
                    gt_poly[:, 2:] *= crop_size

                pred_poly = det[:, :raycast_dim] if det.shape[0] > 0 else np.zeros((0, raycast_dim))
                results.append(dict(pred_polys=pred_poly, gt_polys=gt_poly, imgsz=crop_size))
    return results


def polygons_to_masks(polys, imgsz, n_rays):
    cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    masks = []
    for p in polys:
        cx, cy = p[0], p[1]
        rays = p[2:2 + n_rays]
        if not np.isfinite(rays).all():
            masks.append(np.zeros((imgsz, imgsz), dtype=np.uint8))
            continue
        vx = cx + rays * cos
        vy = cy + rays * sin
        pts = np.stack([vx, vy], axis=-1).clip(-32768, 32767).astype(np.int32).reshape(-1, 1, 2)
        m = np.zeros((imgsz, imgsz), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        masks.append(m)
    return masks


def mask_iou_matrix(pred_masks, gt_masks):
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)))
    ps = np.stack(pred_masks).reshape(len(pred_masks), -1).astype(np.float64)
    gs = np.stack(gt_masks).reshape(len(gt_masks), -1).astype(np.float64)
    inter = ps @ gs.T
    union = ps.sum(1, keepdims=True) + gs.sum(1, keepdims=True).T - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--dataset', required=True, choices=['PUMA', 'MoNuSAC', 'PanopTILs'])
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.49)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)
    model = load_model(args.weights, device)
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 256)
    n_rays = training_args.get('n_rays', 64)
    raycast_dim = 2 + n_rays

    _const.configure_rays(n_rays)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'{data_dir}')

    npz_files = list(data_dir.glob('*.npz'))
    print(f'Test tiles: {len(npz_files)} ({args.dataset})')

    dataset = RayCastTileDataset(data_dir=str(data_dir), crop_size=crop_size, augment=False)
    dl = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=_collate_fn)

    t0 = time.perf_counter()
    results = run_inference(model, dl, device, args.conf, raycast_dim)
    elapsed = time.perf_counter() - t0
    n_pred = sum(len(r['pred_polys']) for r in results)
    n_gt = sum(len(r['gt_polys']) for r in results)
    print(f'Inference: {len(results)} imgs in {elapsed:.1f}s ({elapsed/len(results)*1000:.1f}ms/img), {n_pred} preds, {n_gt} GT')

    # Binary metrics
    aji_scores = []
    bpq_scores = []
    tp_f1 = 0; fp_f1 = 0; fn_f1 = 0
    for i, r in enumerate(results):
        gt_masks = polygons_to_masks(r['gt_polys'], r['imgsz'], n_rays)
        pred_masks = polygons_to_masks(r['pred_polys'], r['imgsz'], n_rays)
        if pred_masks:
            pred_masks = resolve_mask_overlaps(pred_masks)

        aji_scores.append(compute_aji(pred_masks, gt_masks))

        pb = np.stack(pred_masks).max(0).astype(np.uint8) if pred_masks else np.zeros((r['imgsz'], r['imgsz']), dtype=np.uint8)
        gb = np.stack(gt_masks).max(0).astype(np.uint8) if gt_masks else np.zeros((r['imgsz'], r['imgsz']), dtype=np.uint8)

        iou_mat = mask_iou_matrix([pb], [gb])
        if len(pred_masks) and len(gt_masks):
            ri, ci = linear_sum_assignment(-iou_mat)
            valid = iou_mat[ri, ci] >= 0.5
            tp = valid.sum()
            sq = iou_mat[ri[valid], ci[valid]].mean() if tp else 0
            dq = tp / (tp + 0.5*(len(pred_masks)-tp) + 0.5*(len(gt_masks)-tp)) if (tp+len(pred_masks)+len(gt_masks)) else 0
            bpq_scores.append(sq * dq)
        else:
            bpq_scores.append(0.0)

        n_pred = len(r['pred_polys'])
        n_gt = len(r['gt_polys'])
        if n_pred and n_gt:
            dist = np.linalg.norm(r['pred_polys'][:, :2][:, None] - r['gt_polys'][:, :2][None, :], axis=2)
            ri, ci = linear_sum_assignment(dist)
            t = int((dist[ri, ci] <= 12).sum())
        else:
            t = 0
        tp_f1 += t; fp_f1 += n_pred - t; fn_f1 += n_gt - t

        if (i+1) % 500 == 0:
            print(f'  {i+1}/{len(results)}', flush=True)

    aji = np.mean(aji_scores)
    bpq = np.mean(bpq_scores)
    prec = tp_f1 / (tp_f1+fp_f1) if (tp_f1+fp_f1) else 0
    rec = tp_f1 / (tp_f1+fn_f1) if (tp_f1+fn_f1) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0

    print(f'\n{"="*50}')
    print(f'  {args.dataset} Zero-Shot (binary)')
    print(f'{"="*50}')
    print(f'  AJI:      {aji:.4f}')
    print(f'  bPQ:      {bpq:.4f}')
    print(f'  F1:       {f1:.4f}')
    print(f'  Precision:{prec:.4f}')
    print(f'  Recall:   {rec:.4f}')
    print(f'{"="*50}')


def _collate_fn(batch):
    import torch as _torch
    images = _torch.stack([item[0] for item in batch])
    labels_list = [item[1] for item in batch]
    target_list = []
    for bi, labels in enumerate(labels_list):
        if labels.shape[0] == 0: continue
        bc = np.full((labels.shape[0], 1), bi, dtype=np.float32)
        target_list.append(np.concatenate([bc, labels], axis=1))
    if target_list:
        targets = _torch.from_numpy(np.concatenate(target_list, 0))
    else:
        w = next((lbl.shape[1] for lbl in labels_list if lbl.ndim==2 and lbl.shape[1]>0), 35)
        targets = _torch.zeros((0, 1+w), dtype=_torch.float32)
    return dict(img=images, batch_idx=targets[:,0], cls=targets[:,1], bboxes=targets[:,2:])


if __name__ == '__main__':
    main()
