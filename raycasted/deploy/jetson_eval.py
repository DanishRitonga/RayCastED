"""Standalone TensorRT eval for Jetson. Dependencies: numpy, tensorrt, pycuda, scipy, cv2."""
import argparse, json, time, struct, glob, os
from pathlib import Path
import numpy as np

def load_meta(path): 
    with open(path) as f: return json.load(f)

def load_engine(path):
    import tensorrt as trt
    import pycuda.autoinit, pycuda.driver as cuda
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, 'rb') as f: engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    stream = cuda.Stream()
    n_io = engine.num_io_tensors
    bufs = {}
    for i in range(n_io):
        name = engine.get_tensor_name(i)
        shp = engine.get_tensor_shape(name)
        dt = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16}[engine.get_tensor_dtype(name)]
        sz = int(np.prod(shp))
        h = cuda.pagelocked_empty(sz, dt)
        d = cuda.mem_alloc(h.nbytes)
        if engine.get_tensor_mode(name) == trt.TensorMode.INPUT:
            inp = {'host': h, 'device': d, 'shape': shp}
        else:
            bufs[name] = {'host': h, 'device': d, 'shape': shp}
    return engine, ctx, stream, inp, bufs, cuda

def build_anchor_grid(strides, imgsz):
    ax, ay, st = [], [], []
    for s in strides:
        f = int(imgsz / s)
        gx, gy = np.meshgrid(np.arange(f), np.arange(f))
        ax.append((gx + 0.5).flatten().astype(np.float32))
        ay.append((gy + 0.5).flatten().astype(np.float32))
        st.append(np.full(f * f, s, dtype=np.float32))
    return np.concatenate(ax), np.concatenate(ay), np.concatenate(st)

def postprocess(outputs, meta, conf_th=0.5):
    boxes = outputs['boxes'].reshape(-1, 2 + meta['n_rays'])
    hierarchical = meta.get('hierarchical_cls', False)
    nc = meta['nc']
    n_rays = meta['n_rays']
    imgsz = meta['imgsz']
    strides = meta['strides']
    if hierarchical and 'binary' in outputs and 'class' in outputs:
        binary_raw = outputs['binary'].reshape(-1)
        class_raw = outputs['class'].reshape(-1, nc)
        xy = 1.0 / (1.0 + np.exp(-boxes[:, :2]))
        rays = np.log1p(np.exp(boxes[:, 2:]))
        bs = 1.0 / (1.0 + np.exp(-binary_raw))
        cs = 1.0 / (1.0 + np.exp(-class_raw))
        scores = bs * cs.max(axis=1)
        cls_idx = cs.argmax(axis=1)
        hard_gate = bs < meta.get('binary_threshold', 0.01)
        scores[hard_gate] = 0.0
    else:
        scores_raw = outputs.get('scores', outputs.get('class')).reshape(-1, nc)
        xy = 1.0 / (1.0 + np.exp(-boxes[:, :2]))
        rays = np.log1p(np.exp(boxes[:, 2:]))
        cs = 1.0 / (1.0 + np.exp(-scores_raw))
        scores = cs.max(axis=1)
        cls_idx = cs.argmax(axis=1)
    ax, ay, st = build_anchor_grid(strides, imgsz)
    cx = (xy[:, 0] * 2.0 - 0.5 + ax) * st
    cy = (xy[:, 1] * 2.0 - 0.5 + ay) * st
    rays_px = rays * imgsz
    mask = scores > conf_th
    n_det = int(mask.sum())
    if n_det == 0:
        return np.zeros((0, 2 + n_rays + 2), dtype=np.float32)
    det = np.zeros((n_det, 2 + n_rays + 2), dtype=np.float32)
    det[:, 0] = cx[mask]
    det[:, 1] = cy[mask]
    det[:, 2:2 + n_rays] = rays_px[mask]
    det[:, 2 + n_rays] = scores[mask]
    det[:, 2 + n_rays + 1] = cls_idx[mask]
    return det

def infer(ctx, image_np):
    engine, context, stream, inp, buf, cuda = ctx
    blob = np.ascontiguousarray(image_np.astype(np.float32).transpose(2, 0, 1)[np.newaxis])
    np.copyto(inp['host'], blob.ravel())
    cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    context.execute_async_v3(stream.handle)
    for b in buf.values():
        cuda.memcpy_dtoh_async(b['host'], b['device'], stream)
    stream.synchronize()
    return {name: buf[name]['host'].reshape(buf[name]['shape']) for name in buf}

def mask_iou_matrix(pred_masks, gt_masks):
    n_pred, n_gt = len(pred_masks), len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)
    ps = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gs = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)
    inter = ps @ gs.T
    pa = ps.sum(axis=1, keepdims=True)
    ga = gs.sum(axis=1, keepdims=True)
    union = pa + ga.T - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

from scipy.optimize import linear_sum_assignment

def compute_metrics(results, n_rays):
    aji_scores = []
    bpq_scores = []
    f1_tp, f1_fp, f1_fn = 0, 0, 0
    for r in results:
        pred = r['pred']
        gt_p = r['gt_p']
        gt_c = r['gt_c']
        imgsz = r['imgsz']
        # rasterize
        import cv2
        pred_masks = rasterize(pred, n_rays, imgsz)
        gt_masks = rasterize(gt_p, n_rays, imgsz)
        # resolve overlaps
        if pred_masks:
            pred_masks = resolve_overlaps(pred_masks)
        # AJI
        aji_scores.append(compute_aji(pred_masks, gt_masks))
        # bPQ (foreground)
        pred_bin = np.stack(pred_masks).max(axis=0).astype(np.uint8) if pred_masks else np.zeros((imgsz, imgsz), dtype=np.uint8)
        gt_bin = np.stack(gt_masks).max(axis=0).astype(np.uint8) if gt_masks else np.zeros((imgsz, imgsz), dtype=np.uint8)
        iou_mat = mask_iou_matrix([pred_bin], [gt_bin])
        row, col = linear_sum_assignment(-iou_mat)
        valid = iou_mat[row, col] >= 0.5
        tp = valid.sum()
        bpq = (iou_mat[row[valid], col[valid]].mean() * tp / (tp + 0.5 * (1 - tp) + 0.5 * (len(gt_masks) - tp))) if len(gt_masks) and len(pred_masks) else 0.0
        bpq_scores.append(bpq)
        # F1 centroid
        n_pred, n_gt = len(pred), len(gt_p)
        if n_pred > 0 and n_gt > 0:
            dist = np.linalg.norm(pred[:, :2][:, None] - gt_p[:, :2][None, :], axis=2)
            ri, ci = linear_sum_assignment(dist)
            tp = int((dist[ri, ci] <= 12).sum())
        else:
            tp = 0
        f1_tp += tp
        f1_fp += n_pred - tp
        f1_fn += n_gt - tp
    aji = np.mean(aji_scores) if aji_scores else 0
    bpq = np.mean(bpq_scores) if bpq_scores else 0
    prec = f1_tp / (f1_tp + f1_fp) if (f1_tp + f1_fp) else 0
    rec = f1_tp / (f1_tp + f1_fn) if (f1_tp + f1_fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {'aji': aji, 'bpq': bpq, 'f1': f1, 'prec': prec, 'rec': rec}

def rasterize(dets, n_rays, imgsz):
    import cv2
    cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    masks = []
    for det in dets:
        cx, cy = det[0], det[1]
        rays = det[2:2 + n_rays]
        if not np.isfinite(rays).all():
            masks.append(np.zeros((imgsz, imgsz), dtype=np.uint8))
            continue
        vx = cx + rays * cos
        vy = cy + rays * sin
        pts = np.stack([vx, vy], axis=-1).clip(-32768, 32767).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((imgsz, imgsz), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
    return masks

def resolve_overlaps(masks):
    if len(masks) <= 1: return masks
    areas = np.array([m.sum() for m in masks])
    order = np.argsort(-areas)
    composite = np.zeros_like(masks[0], dtype=np.int32)
    result = [None] * len(masks)
    for idx in order:
        m = masks[idx]
        result[idx] = (m.astype(np.int32) * (idx + 1))
        composite = np.maximum(composite, result[idx])
    for i in range(len(masks)):
        result[i] = (composite == (i + 1)).astype(np.uint8)
    return result

def compute_aji(pred_masks, gt_masks):
    from scipy.optimize import linear_sum_assignment
    if not pred_masks or not gt_masks: return 0.0
    iou = mask_iou_matrix(pred_masks, gt_masks)
    ri, ci = linear_sum_assignment(-iou)
    valid = iou[ri, ci] >= 0.5
    if not valid.any(): return 0.0
    matched_iou = iou[ri[valid], ci[valid]]
    return matched_iou.mean()

def recall_diag(results, n_rays):
    areas_matched, areas_unmatched = [], []
    for r in results:
        pred, gt_p, gt_c = r['pred'], r['gt_p'], r['gt_c']
        n_gt = len(gt_p)
        if n_gt == 0: continue
        matched = np.zeros(n_gt, dtype=bool)
        if len(pred) > 0:
            dist = np.linalg.norm(pred[:, :2][:, None] - gt_p[:, :2][None, :], axis=2)
            ri, ci = linear_sum_assignment(dist)
            matched[ri[(dist[ri, ci] <= 12)]] = True
        for j in range(n_gt):
            a = shoelace_area(gt_p[j], n_rays)
            if matched[j]: areas_matched.append(a)
            else: areas_unmatched.append(a)
    all_a = np.array(areas_matched + areas_unmatched)
    if len(all_a) == 0: return
    p33, p67 = np.percentile(all_a, [33, 67])
    for label, lo, hi in [('Small', 0, p33), ('Medium', p33, p67), ('Large', p67, float('inf'))]:
        t = sum(1 for a in all_a if lo <= a < hi)
        m = sum(1 for a in areas_matched if lo <= a < hi)
        print(f'  {label}: {m}/{t} ({m/t:.1%})' if t else f'  {label}: 0/0')

def shoelace_area(poly, n_rays):
    cx, cy, rays = poly[0], poly[1], poly[2:2 + n_rays]
    cos = np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    sin = np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False))
    vx, vy = cx + rays * cos, cy + rays * sin
    return 0.5 * abs(np.dot(vx, np.roll(vy, 1)) - np.dot(vy, np.roll(vx, 1)))

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--engine'); g.add_argument('--onnx')
    p.add_argument('--meta', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--max-images', type=int, default=0)
    args = p.parse_args()
    meta = load_meta(args.meta)
    n_rays = meta['n_rays']
    imgsz = meta['imgsz']
    # Load backend
    if args.engine:
        ctx = load_engine(args.engine)
        print(f'Engine: {args.engine}')
    else:
        import onnxruntime as ort
        session = ort.InferenceSession(args.onnx)
        print(f'ONNX: {args.onnx}')
    # Collect npz files
    files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    if args.max_images: files = files[:args.max_images]
    print(f'{len(files)} tiles')
    results = []
    npred, ngt = 0, 0
    t0 = time.perf_counter()
    for fi, path in enumerate(files):
        data = dict(np.load(path, allow_pickle=True))
        img = data['image']
        ann = data['annotations']
        if img.dtype == np.uint8: img = img.astype(np.float32) / 255.0
        if args.engine:
            raw = infer(ctx, img)
        else:
            blob = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis].astype(np.float32))
            raw_ort = session.run(None, {'images': blob})
            names = [n.name for n in session.get_outputs()]
            raw = {names[j]: raw_ort[j] for j in range(len(names))}
        det = postprocess(raw, meta, args.conf)
        npred += det.shape[0]
        ngt += ann.shape[0]
        gt = ann.copy()
        if gt.shape[0]:
            gt[:, 0] *= imgsz; gt[:, 1] *= imgsz; gt[:, 2:] *= imgsz
        results.append({'pred': det, 'gt_p': gt[:, 1:] if gt.shape[0] else np.zeros((0, 2 + n_rays)),
                        'gt_c': gt[:, 0].astype(int) if gt.shape[0] else np.array([]), 'imgsz': imgsz})
        if (fi + 1) % 500 == 0:
            print(f'  {fi+1}/{len(files)} ({time.perf_counter() - t0:.1f}s)', flush=True)
    elapsed = time.perf_counter() - t0
    ms = elapsed / max(len(results), 1) * 1000
    print(f'Done: {len(results)} in {elapsed:.1f}s ({ms:.1f}ms/img), preds={npred}, gt={ngt}')
    m = compute_metrics(results, n_rays)
    print(f'\nAJI={m[\"aji\"]:.4f}  bPQ={m[\"bpq\"]:.4f}  F1={m[\"f1\"]:.4f}  Pr={m[\"prec\"]:.4f}  Re={m[\"rec\"]:.4f}  ms/img={ms:.1f}')
    print('\nSize bins:')
    recall_diag(results, n_rays)

if __name__ == '__main__':
    main()
