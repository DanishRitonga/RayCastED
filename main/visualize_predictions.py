r"""RayCastED — Visual Test Runner.

Runs inference on sample tiles and renders GT + predicted polygons as PNGs.
Works with both PyTorch (desktop) and TensorRT (Jetson).

USAGE:
  # Desktop (PyTorch)
  uv run python main/visualize_predictions.py \
      --weights docs/runs/v1.1/best.pt --data-dir output/visualTest --conf 0.5

  # Jetson (TensorRT)
  uv run python main/visualize_predictions.py \
      --engine ./onnx/v1.engine --meta ./onnx/v1.meta.json \
      --data-dir output/visualTest --conf 0.5
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

CLS_COLORS = [
    (0, 255, 0),      # Neoplastic — green
    (255, 0, 0),      # Inflammatory — blue (BGR)
    (0, 0, 255),      # Connective — red
    (0, 255, 255),    # Dead — yellow
    (255, 0, 255),    # Epithelial — magenta
]

CLS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']


def _configure_rays(n_rays: int):
    angles = 2.0 * np.pi * np.arange(n_rays, dtype=np.float64) / n_rays
    global RAY_COS, RAY_SIN
    RAY_COS = np.cos(angles).astype(np.float32)
    RAY_SIN = np.sin(angles).astype(np.float32)


def draw_polygon(img_bgr, cx, cy, rays, color, thickness=2):
    vx = cx + rays * RAY_COS
    vy = cy + rays * RAY_SIN
    pts = np.stack([vx, vy], axis=-1)
    pts = np.clip(pts, -32768, 32767).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], True, color, thickness)


def draw_gt(img_bgr, labels):
    for lbl in labels:
        cls_id = int(lbl[0])
        cx, cy = lbl[1], lbl[2]
        rays = lbl[3:3 + len(RAY_COS)]
        draw_polygon(img_bgr, cx, cy, rays, CLS_COLORS[cls_id % 5], thickness=2)


def draw_pred(img_bgr, det, raycast_dim):
    for d in det:
        cx, cy = d[0], d[1]
        rays = d[2:raycast_dim]
        cls_id = int(d[raycast_dim + 1])
        if cls_id >= len(CLS_COLORS):
            cls_id = 0
        draw_polygon(img_bgr, cx, cy, rays, CLS_COLORS[cls_id], thickness=1)


def create_legend(img_h):
    legend = np.zeros((img_h, img_h, 3), dtype=np.uint8)
    y = 20
    for i, (name, color) in enumerate(zip(CLS_NAMES, CLS_COLORS)):
        cv2.putText(legend, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
    return legend


def postprocess(boxes_raw, binary_raw, class_raw, strides, imgsz,
                conf_threshold=0.20, n_rays=64):
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
        max_scores[binary_scores < 0.01] = 0.0
    else:
        max_scores = cls_scores.max(axis=1)
        cls_idx = cls_scores.argmax(axis=1)

    all_ax, all_ay, all_s = [], [], []
    for stride in strides:
        fs = int(imgsz / stride)
        gx, gy = np.meshgrid(np.arange(fs), np.arange(fs))
        all_ax.append((gx + 0.5).flatten().astype(np.float32))
        all_ay.append((gy + 0.5).flatten().astype(np.float32))
        all_s.append(np.full(fs * fs, stride, dtype=np.float32))
    anchor_x = np.concatenate(all_ax)
    anchor_y = np.concatenate(all_ay)
    stride_arr = np.concatenate(all_s)

    cx = (xy_offset[:, 0] * 2.0 - 0.5 + anchor_x) * stride_arr
    cy = (xy_offset[:, 1] * 2.0 - 0.5 + anchor_y) * stride_arr
    rays_px = rays * imgsz

    mask = max_scores > conf_threshold
    indices = np.where(mask)[0]
    n_det = len(indices)
    if n_det == 0:
        return np.zeros((0, raycast_dim + 2), dtype=np.float32)
    if n_det > 100:
        topk = np.argsort(-max_scores[indices])[:100]
        indices = indices[topk]
        n_det = 100

    det = np.zeros((n_det, raycast_dim + 2), dtype=np.float32)
    det[:, 0] = cx[indices]
    det[:, 1] = cy[indices]
    det[:, 2:2 + n_rays] = rays_px[indices]
    det[:, raycast_dim] = max_scores[indices]
    det[:, raycast_dim + 1] = cls_idx[indices]
    return det


def load_trt_engine(engine_path: str):
    import pycuda.autoinit  # noqa: F401
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


def main():
    parser = argparse.ArgumentParser(description='Visualize RayCastED predictions')
    parser.add_argument('--data-dir', required=True, help='Directory of .npz tiles')
    parser.add_argument('--conf', type=float, default=0.5)

    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument('--weights', help='PyTorch checkpoint')
    backend.add_argument('--engine', help='TensorRT engine (--meta required)')
    backend.add_argument('--onnx', help='ONNX model (--meta required)')
    parser.add_argument('--meta', help='Metadata JSON (required with --engine/--onnx)')
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).glob('*.npz'))
    if not files:
        print(f'No .npz files found in {args.data_dir}')
        return

    if args.weights:
        from raycasted.model.register import register_raycast_head
        import torch

        register_raycast_head()
        model = torch.load(args.weights, map_location='cpu', weights_only=False)
        if isinstance(model, dict):
            model = model.get('model') or model.get('ema') or model
        model.eval()
        head = model.model[-1]
        n_rays = head.n_rays
        imgsz = 256
        strides = [4, 8, 16]
        use_trt = False
        use_onnx = False
        print(f'Loaded PyTorch model: n_rays={n_rays}')
    else:
        if not args.meta:
            parser.error('--meta required with --engine/--onnx')
        with open(args.meta) as f:
            meta = json.load(f)
        n_rays = meta['n_rays']
        imgsz = meta['imgsz']
        strides = meta.get('strides', [4, 8, 16])
        use_trt = args.engine is not None
        use_onnx = args.onnx is not None
        if use_trt:
            trt_ctx = load_trt_engine(args.engine)
            print(f'Loaded TRT engine: n_rays={n_rays} imgsz={imgsz}')
        else:
            import onnxruntime as ort

            ort_session = ort.InferenceSession(args.onnx)
            print(f'Loaded ONNX model: n_rays={n_rays} imgsz={imgsz}')

    _configure_rays(n_rays)
    raycast_dim = 2 + n_rays
    out_dir = Path(args.data_dir) / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    for f_path in files:
        data = dict(np.load(f_path, allow_pickle=True))
        image = data['image']
        labels = data['annotations']

        t0 = time.perf_counter()

        if use_trt:
            import pycuda.driver as cuda

            blob = image.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis]
            engine, context, stream, d_input, buffers, cuda_mod = trt_ctx
            np.copyto(d_input['host'], blob.ravel())
            cuda.memcpy_htod_async(d_input['device'], d_input['host'], stream)
            context.execute_async_v3(stream.handle)
            for _, buf in buffers.items():
                cuda.memcpy_dtoh_async(buf['host'], buf['device'], stream)
            stream.synchronize()

            outputs = {}
            for name, buf in buffers.items():
                outputs[name] = buf['host'].reshape(buf['shape'])
            for k in outputs:
                if outputs[k].ndim == 3:
                    outputs[k] = outputs[k][0].T

            det = postprocess(outputs['boxes'], None,
                              outputs.get('scores', outputs.get('class')),
                              strides, imgsz, args.conf, n_rays=n_rays)
        elif use_onnx:
            blob = image.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis]
            ort_outs = ort_session.run(None, {'images': blob})
            det = postprocess(ort_outs[0][0].T, None, ort_outs[1][0].T,
                              strides, imgsz, args.conf, n_rays=n_rays)
        else:
            import torch

            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                det = model(tensor)[0]
            if det is None or len(det) == 0:
                det = np.zeros((0, raycast_dim + 2), dtype=np.float32)
            elif hasattr(det, 'cpu'):
                det = det.cpu().numpy()

        elapsed = (time.perf_counter() - t0) * 1000

        gt_img = image.copy()
        draw_gt(gt_img, labels)

        pred_img = image.copy()
        draw_pred(pred_img, det, raycast_dim)

        img_h = image.shape[0]
        legend = create_legend(200)
        if img_h > legend.shape[0]:
            pad = np.zeros((img_h - legend.shape[0], legend.shape[1], 3), dtype=np.uint8)
            legend = np.vstack([legend, pad])
        legend[10:30, :] = (255, 255, 255)
        cv2.putText(legend, f'Pred: {det.shape[0]}', (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(legend, f'GT: {labels.shape[0]}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(legend, f'{elapsed:.0f}ms', (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        combined = np.hstack([gt_img, pred_img, legend])

        img_out_dir = out_dir / f_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_out_dir / 'gt.png'), gt_img)
        cv2.imwrite(str(img_out_dir / 'pred.png'), pred_img)
        cv2.imwrite(str(img_out_dir / 'combined.png'), combined)

        stem = f_path.stem
        print(f'{stem:<50} GT={labels.shape[0]:>3} Pred={det.shape[0]:>3} {elapsed:.1f}ms')

    print(f'\nSaved to {out_dir}/')


if __name__ == '__main__':
    main()
