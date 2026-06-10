r"""RayCastED — Standalone Jetson Evaluation.

Self-contained eval script for NVIDIA Jetson Orin Nano.
No PyTorch, no ultralytics, no raycasted package needed.

DEPENDENCIES (pre-installed on JetPack):
  - numpy, tensorrt, pycuda

DEPENDENCIES (pip install):
  - scipy (linear_sum_assignment for metrics)
  - opencv-python (mask rasterization)

USAGE:
  python main/eval_jetson.py \
      --engine raycasted.256.engine \
      --meta raycasted.256.meta.json \
      --data-dir output/pannuke_64/transformed/test \
      --conf 0.5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ============================================================
# Ray geometry (inlined from constants.py — avoids torch import)
# ============================================================

def _configure_rays(n_rays: int):
    """Inline version of configure_rays(n_rays)."""
    angles = 2.0 * np.pi * np.arange(n_rays, dtype=np.float64) / n_rays
    global RAY_COS, RAY_SIN
    RAY_COS = np.cos(angles).astype(np.float32)
    RAY_SIN = np.sin(angles).astype(np.float32)


def _polygons_to_masks(detections, img_h, img_w):
    """Rasterize polygons to binary masks (cv2-based)."""
    import cv2

    masks = []
    for det in detections:
        cx, cy = det[0], det[1]
        rays = det[2:]
        if not np.isfinite(rays).all() or not np.isfinite(cx) or not np.isfinite(cy):
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
        vx = cx + rays * RAY_COS
        vy = cy + rays * RAY_SIN
        pts = np.stack([vx, vy], axis=-1)
        pts = np.clip(pts, -32768, 32767).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
    return masks


def _polygon_area(poly):
    cx, cy = poly[0], poly[1]
    rays = poly[2:]
    if len(rays) < 3:
        return 0.0
    vx = cx + rays * RAY_COS
    vy = cy + rays * RAY_SIN
    return 0.5 * abs(np.dot(vx, np.roll(vy, 1)) - np.dot(vy, np.roll(vx, 1)))


# ============================================================
# Post-processing (from postprocess.py — numpy only)
# ============================================================

def _build_anchor_grid(strides, imgsz):
    all_ax, all_ay, all_s = [], [], []
    for stride in strides:
        feat_size = int(imgsz / stride)
        gx, gy = np.meshgrid(np.arange(feat_size), np.arange(feat_size))
        all_ax.append((gx + 0.5).flatten().astype(np.float32))
        all_ay.append((gy + 0.5).flatten().astype(np.float32))
        all_s.append(np.full(feat_size * feat_size, stride, dtype=np.float32))
    return np.concatenate(all_ax), np.concatenate(all_ay), np.concatenate(all_s)


def postprocess(boxes_raw, binary_raw, class_raw, strides, imgsz,
                conf_threshold=0.20, binary_threshold=0.01, n_rays=64):
    """Post-process raw ONNX output to polygon detections."""
    boxes_raw = boxes_raw.squeeze(0) if boxes_raw.ndim == 4 else boxes_raw
    class_raw = class_raw.squeeze(0) if class_raw.ndim >= 3 else class_raw
    if binary_raw is not None:
        binary_raw = binary_raw.squeeze() if binary_raw.ndim >= 3 else binary_raw

    raycast_dim = 2 + n_rays
    xy_offset = 1.0 / (1.0 + np.exp(-boxes_raw[:, :2]))
    rays = np.log1p(np.exp(boxes_raw[:, 2:]))

    cls_scores = 1.0 / (1.0 + np.exp(-class_raw))

    if binary_raw is not None:
        binary_scores = 1.0 / (1.0 + np.exp(-binary_raw))
        combined = binary_scores.reshape(-1, 1) * cls_scores.max(axis=1, keepdims=True)
        cls_idx = cls_scores.argmax(axis=1)
        max_scores = combined.ravel()
        max_scores[binary_scores < binary_threshold] = 0.0
    else:
        max_scores = cls_scores.max(axis=1)
        cls_idx = cls_scores.argmax(axis=1)

    anchor_x, anchor_y, stride_arr = _build_anchor_grid(strides, imgsz)
    cx = (xy_offset[:, 0] * 2.0 - 0.5 + anchor_x) * stride_arr
    cy = (xy_offset[:, 1] * 2.0 - 0.5 + anchor_y) * stride_arr
    rays_px = rays * imgsz

    mask = max_scores > conf_threshold
    n_det = int(mask.sum())
    if n_det == 0:
        return np.zeros((0, raycast_dim + 2), dtype=np.float32)

    det = np.zeros((n_det, raycast_dim + 2), dtype=np.float32)
    det[:, 0] = cx[mask]
    det[:, 1] = cy[mask]
    det[:, 2:2 + n_rays] = rays_px[mask]
    det[:, raycast_dim] = max_scores[mask]
    det[:, raycast_dim + 1] = cls_idx[mask]
    return det


# ============================================================
# TensorRT inference
# ============================================================

def load_trt_engine(engine_path: str):
    import pycuda.autoinit  # noqa
    import pycuda.driver as cuda
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    stream = cuda.Stream()

    d_input = None
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        np_dtype = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16}.get(dtype, np.float32)
        size = int(np.prod(shape))
        h_mem = cuda.pagelocked_empty(size, np_dtype)
        d_mem = cuda.mem_alloc(h_mem.nbytes)
        context.set_tensor_address(name, int(d_mem))
        trt_input = getattr(trt, 'TensorIOMode', getattr(trt, 'TensorMode', None)).INPUT
        if engine.get_tensor_mode(name) == trt_input:
            d_input = {'name': name, 'host': h_mem, 'device': d_mem, 'shape': shape}
        else:
            buffers[name] = {'host': h_mem, 'device': d_mem, 'shape': shape}
    return engine, context, stream, d_input, buffers, cuda


def run_trt(ctx, blob: np.ndarray):
    engine, context, stream, d_input, buffers, cuda = ctx
    np.copyto(d_input['host'], blob.ravel())
    cuda.memcpy_htod_async(d_input['device'], d_input['host'], stream)
    context.execute_async_v3(stream.handle)
    results = {}
    for name, buf in buffers.items():
        cuda.memcpy_dtoh_async(buf['host'], buf['device'], stream)
    stream.synchronize()
    for name, buf in buffers.items():
        results[name] = buf['host'].reshape(buf['shape'])
    return results


# ============================================================
# Mask IoU metrics (inlined from eval_pannuke.py — numpy+scipy)
# ============================================================

def _mask_iou_matrix(pred_masks, gt_masks):
    from scipy.optimize import linear_sum_assignment

    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)
    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)
    intersection = pred_stack @ gt_stack.T
    pred_area = pred_stack.sum(axis=1, keepdims=True)
    gt_area = gt_stack.sum(axis=1, keepdims=True)
    union = pred_area + gt_area.T - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)


def _compute_pq(pred_masks, gt_masks, iou_threshold=0.5, mask=None):
    from scipy.optimize import linear_sum_assignment

    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0
    if mask is not None:
        pred_masks = [m & mask for m in pred_masks]
        gt_masks = [m & mask for m in gt_masks]
    iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold
    tp = valid.sum()
    fp = n_pred - tp
    fn = n_gt - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0
    return float(sq * dq), float(sq), float(dq)


def _compute_aji(pred_masks, gt_masks):
    from scipy.optimize import linear_sum_assignment

    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_gt == 0:
        return 0.0
    if n_pred == 0:
        return 0.0

    iou = _mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou)
    valid = iou[row_ind, col_ind] >= 0.5
    match_pred = set(row_ind[valid].tolist())
    match_gt = set(col_ind[valid].tolist())

    total_inter = 0.0
    total_union = 0.0
    for r, c in zip(row_ind[valid], col_ind[valid]):
        p = pred_masks[r].astype(np.float64)
        g = gt_masks[c].astype(np.float64)
        total_inter += (p * g).sum()
        total_union += (p + g - p * g).sum()

    for i in range(n_pred):
        if i not in match_pred:
            total_union += pred_masks[i].astype(np.float64).sum()
    for j in range(n_gt):
        if j not in match_gt:
            total_union += gt_masks[j].astype(np.float64).sum()

    return float(total_inter / total_union) if total_union > 0 else 0.0


def _resolve_overlaps(masks):
    if len(masks) <= 1:
        return masks
    overlap = np.stack(masks).sum(axis=0) > 1
    if not overlap.any():
        return masks
    areas = np.array([m.sum() for m in masks])
    order = np.argsort(-areas)
    resolved = [m.copy() for m in masks]
    for i_idx, i in enumerate(order):
        for j in order[i_idx + 1:]:
            resolved[j] = resolved[j] & ~resolved[i]
    return resolved


def _compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


# ============================================================
# Main eval loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Standalone Jetson Evaluation')
    parser.add_argument('--engine', required=True, help='TensorRT engine')
    parser.add_argument('--meta', required=True, help='Metadata JSON')
    parser.add_argument('--data-dir', required=True, help='Directory of .npz tiles')
    parser.add_argument('--conf', type=float, default=0.50)
    parser.add_argument('--max-images', type=int, default=0)
    args = parser.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    imgsz = meta['imgsz']
    n_rays = meta['n_rays']
    nc = meta.get('nc', 5)
    strides = meta.get('strides', [4, 8, 16])
    hierarchical = meta.get('hierarchical_cls', False)
    binary_threshold = meta.get('binary_threshold', 0.01)
    raycast_dim = 2 + n_rays

    _configure_rays(n_rays)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob('*.npz'))
    if args.max_images:
        files = files[:args.max_images]

    print(f'Loading engine: {args.engine}')
    trt_ctx = load_trt_engine(args.engine)

    bpq_scores = []
    aji_scores = []
    class_pq = {c: [] for c in range(nc)}
    centroid_tp = 0
    centroid_fp = 0
    centroid_fn = 0
    iou_thresholds = sorted(set(round(x, 2) for x in np.arange(0.5, 1.0, 0.05)))
    class_set = set()
    ap_stats = {}
    all_results = []
    n_pred_total = 0
    n_gt_total = 0

    t_start = time.perf_counter()
    print(f'Running {len(files)} images (conf={args.conf})...')

    for i, f_path in enumerate(files):
        data = dict(np.load(f_path, allow_pickle=True))
        image = data['image'].astype(np.float32) / 255.0  # uint8→float [0,1]
        labels = data['annotations']

        # TRT inference — model expects CHW float32 [0,1]
        blob = image.transpose(2, 0, 1)[np.newaxis]
        outputs = run_trt(trt_ctx, blob)

        # TRT returns [1, C, N] — transpose to [N, C] for postprocess
        for k in outputs:
            if outputs[k].ndim == 3:
                outputs[k] = outputs[k][0].T

        # Post-process
        if hierarchical and 'binary' in outputs:
            det = postprocess(outputs['boxes'], outputs['binary'], outputs['class'],
                              strides, imgsz, args.conf, binary_threshold, n_rays)
        else:
            scores = outputs.get('scores', outputs.get('class'))
            det = postprocess(outputs['boxes'], None, scores,
                              strides, imgsz, args.conf, n_rays=n_rays)

        # GT — labels are already in pixel coordinates
        gt_poly = labels[:, 1:].copy()
        n_gt = labels.shape[0]

        n_pred_total += det.shape[0]
        n_gt_total += n_gt

        # Rasterize
        gt_masks = _polygons_to_masks(gt_poly, imgsz, imgsz) if n_gt > 0 else []
        pred_masks = _polygons_to_masks(det[:, :raycast_dim], imgsz, imgsz) if det.shape[0] > 0 else []
        if pred_masks:
            pred_masks = _resolve_overlaps(pred_masks)

        # AJI
        aji_scores.append(_compute_aji(pred_masks, gt_masks))

        # bPQ
        if pred_masks:
            pred_bin = np.stack(pred_masks).max(axis=0).astype(np.uint8)
        else:
            pred_bin = np.zeros((imgsz, imgsz), dtype=np.uint8)
        if gt_masks:
            gt_bin = np.stack(gt_masks).max(axis=0).astype(np.uint8)
        else:
            gt_bin = np.zeros((imgsz, imgsz), dtype=np.uint8)
        bpq, _, _ = _compute_pq([pred_bin], [gt_bin])
        bpq_scores.append(bpq)

        # mPQ (per-class)
        for c in range(nc):
            c_pred = [i for i, lbl in enumerate(det[:, raycast_dim + 1].astype(int)) if lbl == c] if det.shape[0] > 0 else []
            c_gt = [i for i, lbl in enumerate(labels[:, 0].astype(int)) if lbl == c]
            pc = [pred_masks[i] for i in c_pred]
            gc = [gt_masks[i] for i in c_gt]
            pq, _, _ = _compute_pq(pc, gc)
            class_pq[c].append(pq)
            class_set.add(c)

        # AP accumulators
        for c in sorted(class_set):
            if c not in ap_stats:
                ap_stats[c] = {t: {'tp': [], 'fp': [], 'conf': [], 'n_gt': 0} for t in iou_thresholds}

        for c in sorted(class_set):
            c_pred_idx = [j for j in range(len(pred_masks))
                          if j < det.shape[0] and int(det[j, raycast_dim + 1]) == c]
            c_gt_idx = [j for j in range(len(gt_masks))
                        if j < n_gt and int(labels[j, 0]) == c]
            c_pm = [pred_masks[j] for j in c_pred_idx]
            c_gm = [gt_masks[j] for j in c_gt_idx]
            c_conf = det[c_pred_idx, raycast_dim] if c_pred_idx else np.array([])

            n_pc = len(c_pm)
            n_gc = len(c_gm)
            for t in iou_thresholds:
                ap_stats[c][t]['n_gt'] += n_gc

            if n_pc > 0 and n_gc > 0:
                iou_mat = _mask_iou_matrix(c_pm, c_gm)
                from scipy.optimize import linear_sum_assignment
                ri, ci = linear_sum_assignment(-iou_mat)
                matched_iou = iou_mat[ri, ci]
            else:
                ri, matched_iou = np.array([], dtype=int), np.array([], dtype=float)

            for t in iou_thresholds:
                matched_pred = set(ri[np.where(matched_iou >= t)[0]].tolist()) if n_gc > 0 else set()
                for pi in range(n_pc):
                    is_tp = pi in matched_pred
                    ap_stats[c][t]['conf'].append(float(c_conf[pi]) if pi < len(c_conf) else 0.0)
                    ap_stats[c][t]['tp'].append(is_tp)
                    ap_stats[c][t]['fp'].append(not is_tp)

        del gt_masks, pred_masks

        if (i + 1) % 500 == 0:
            e = time.perf_counter() - t_start
            print(f'  {i+1}/{len(files)} images ({e:.1f}s)', flush=True)

    elapsed = time.perf_counter() - t_start
    ms_per_img = elapsed / len(files) * 1000
    print(f'  Done: {len(files)} images in {elapsed:.1f}s ({ms_per_img:.1f} ms/img)')
    print(f'  Predictions: {n_pred_total}, GT: {n_gt_total}')

    # Aggregate
    mean_aji = np.mean(aji_scores)
    mean_bpq = np.mean(bpq_scores)
    mpq_values = [np.mean([v for v in class_pq[c] if v > 0]) for c in range(nc) if any(v > 0 for v in class_pq[c])]
    mean_mpq = np.mean(mpq_values) if mpq_values else 0.0

    ap_results = {}
    for t in iou_thresholds:
        aps = []
        for c in sorted(class_set):
            s = ap_stats[c][t]
            if s['n_gt'] == 0:
                continue
            confs = np.array(s['conf'])
            tps = np.array(s['tp'])
            fps = np.array(s['fp'])
            if len(confs) == 0:
                aps.append(0.0)
                continue
            order = np.argsort(-confs)
            tps = tps[order]
            fps = fps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            prec = cum_tp / (cum_tp + cum_fp + 1e-12)
            rec = cum_tp / s['n_gt']
            aps.append(_compute_ap(rec, prec))
        ap_results[t] = {'AP': np.mean(aps) if aps else 0.0}

    ap50 = ap_results.get(0.5, {}).get('AP', 0.0)
    ap50_95 = np.mean([ap_results[t]['AP'] for t in sorted(ap_results.keys())])

    print('\n' + '=' * 60)
    print('Jetson TensorRT Evaluation Results')
    print('=' * 60)
    print(f'{"AJI":<25} {mean_aji:>12.4f}')
    print(f'{"AP@0.5":<25} {ap50:>12.4f}')
    print(f'{"AP@0.5:0.05:0.95":<25} {ap50_95:>12.4f}')
    print(f'{"bPQ":<25} {mean_bpq:>12.4f}')
    print(f'{"mPQ":<25} {mean_mpq:>12.4f}')
    print(f'{"Inference (ms/img)":<25} {ms_per_img:>12.1f}')
    print('=' * 37)


if __name__ == '__main__':
    main()
