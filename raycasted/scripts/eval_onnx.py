"""RayCastED — ONNX-TensorRT Evaluation Runner.

Runs the full eval_pannuke.py pipeline using TensorRT inference.
Works on both x86 (ONNX Runtime) and Jetson (TensorRT).

Usage:
    # On x86 — validate ONNX export
    python -m raycasted.scripts.eval_onnx --onnx export/model.onnx --meta export/model.meta.json
        --data-dir output/pannuke_64/transformed/test --conf 0.5

    # On Jetson — TensorRT engine
    python -m raycasted.scripts.eval_onnx --engine export/model.engine --meta export/model.meta.json
        --data-dir output/pannuke_64/transformed/test --conf 0.5
"""

import argparse
import time
from pathlib import Path

import numpy as np

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.utils import constants as _const
from raycasted.model.metrics import compute_aji, resolve_mask_overlaps
from raycasted.export.postprocess import postprocess_raw_output


def load_ort_session(onnx_path: str):
    import onnxruntime as ort

    return ort.InferenceSession(onnx_path)


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

    buffers = {}
    d_input = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        dtype = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16}.get(dtype, np.float32)
        size = int(np.prod(shape))
        h_mem = cuda.pagelocked_empty(size, dtype)
        d_mem = cuda.mem_alloc(h_mem.nbytes)
        context.set_tensor_address(name, int(d_mem))
        trt_input = getattr(trt, 'TensorIOMode', getattr(trt, 'TensorMode', None)).INPUT

        if engine.get_tensor_mode(name) == trt_input:
            d_input = {'host': h_mem, 'device': d_mem, 'shape': shape}
        else:
            buffers[name] = {'host': h_mem, 'device': d_mem, 'shape': shape}

    return engine, context, stream, d_input, buffers, cuda


def run_ort(session, image: np.ndarray, meta: dict) -> np.ndarray:
    """ONNX Runtime inference — single image."""
    blob = image.astype(np.float32)[np.newaxis]
    outputs = session.run(None, {'images': blob})
    names = session.get_outputs()

    if len(names) >= 3 and 'binary' in [n.name for n in names]:
        idx = {n.name: i for i, n in enumerate(names)}
        return postprocess_raw_output(
            outputs[idx['boxes']],
            outputs[idx['binary']],
            outputs[idx['class']],
            meta['strides'],
            meta['imgsz'],
            meta.get('conf_threshold', 0.20),
            meta.get('binary_threshold', 0.01),
            meta['n_rays'],
        )
    else:
        return postprocess_raw_output(
            outputs[0],
            None,
            outputs[1],
            meta['strides'],
            meta['imgsz'],
            meta.get('conf_threshold', 0.20),
            n_rays=meta['n_rays'],
        )


def run_trt(ctx, blob: np.ndarray) -> dict[str, np.ndarray]:
    """TensorRT inference — single image. ctx = (engine, context, stream, d_input, buffers, cuda)."""
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


def main():
    parser = argparse.ArgumentParser(description='ONNX/TensorRT Evaluation')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--onnx', help='Path to ONNX model')
    group.add_argument('--engine', help='Path to TensorRT engine')
    parser.add_argument('--meta', required=True, help='Path to metadata JSON')
    parser.add_argument('--data-dir', required=True, help='Directory of .npz test tiles')
    parser.add_argument('--batch', type=int, default=1, help='Batch size (Jetson: keep at 1)')
    parser.add_argument('--conf', type=float, default=0.50)
    parser.add_argument('--max-images', type=int, default=0, help='Limit images for quick test')
    args = parser.parse_args()

    import json

    with open(args.meta) as f:
        meta = json.load(f)

    imgsz = meta['imgsz']
    n_rays = meta['n_rays']
    nc = meta.get('nc', 5)
    raycast_dim = 2 + n_rays
    conf_threshold = args.conf
    binary_threshold = meta.get('binary_threshold', 0.01)
    strides = meta.get('strides', [4, 8, 16])
    hierarchical = meta.get('hierarchical_cls', False)

    _const.configure_rays(n_rays)

    dataset = RayCastTileDataset(data_dir=args.data_dir, crop_size=imgsz, augment=False)
    from torch.utils.data import DataLoader

    def collate(batch):
        return batch

    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)

    # Setup inference backend
    use_trt = args.engine is not None
    if use_trt:
        trt_ctx = load_trt_engine(args.engine)
        print(f'Loaded TensorRT engine: {args.engine}')
    else:
        session = load_ort_session(args.onnx)
        print(f'Loaded ONNX model: {args.onnx}')

    results = []
    n_pred_total = 0
    n_gt_total = 0

    print(f'Running inference on {len(dataset)} images (conf={conf_threshold})...')
    t_start = time.perf_counter()

    for i, batch in enumerate(dataloader):
        if args.max_images and i >= args.max_images:
            break

        image, labels, _ = batch[0]
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        image_np = image.astype(np.float32)
        if image_np.ndim == 3 and image_np.shape[-1] == 3:
            image_np = image_np.transpose(2, 0, 1)  # HWC → CHW
        elif image_np.ndim == 3 and image_np.shape[0] == 3:
            pass  # already CHW
        blob = image_np[np.newaxis]

        # Inference
        if use_trt:
            outputs = run_trt(trt_ctx, blob)
        else:
            outputs_ort = session.run(None, {'images': blob})
            names = [n.name for n in session.get_outputs()]
            outputs = {names[j]: outputs_ort[j] for j in range(len(names))}

        for k in outputs:
            if outputs[k].ndim == 3:
                outputs[k] = outputs[k][0].T  # [1, C, N] → [N, C]

        # Postprocess
        if hierarchical and 'binary' in outputs:
            det = postprocess_raw_output(
                outputs['boxes'],
                outputs['binary'],
                outputs['class'],
                strides,
                imgsz,
                conf_threshold,
                binary_threshold,
                n_rays,
            )
        else:
            scores = outputs.get('scores', outputs.get('class'))
            det = postprocess_raw_output(
                outputs['boxes'],
                None,
                scores,
                strides,
                imgsz,
                conf_threshold,
                n_rays=n_rays,
            )

        # GT
        n_gt = labels.shape[0] if labels.ndim >= 2 else 0
        if n_gt > 0:
            gt_poly = labels[:, 1:].copy()
            gt_poly[:, 0] *= imgsz
            gt_poly[:, 1] *= imgsz
            gt_poly[:, 2:] *= imgsz
        else:
            gt_poly = np.zeros((0, 2 + n_rays), dtype=np.float32)

        n_pred_total += det.shape[0]
        n_gt_total += n_gt

        results.append(
            {
                'pred_polys': det[:, :raycast_dim] if det.shape[0] > 0 else np.zeros((0, raycast_dim)),
                'pred_confs': det[:, raycast_dim] if det.shape[0] > 0 else np.array([]),
                'pred_cls': det[:, raycast_dim + 1].astype(int) if det.shape[0] > 0 else np.array([]),
                'gt_polys': gt_poly,
                'gt_cls': labels[:, 0].astype(int)
                if labels.ndim >= 2 and labels.shape[0] > 0
                else np.array([], dtype=int),
                'imgsz': imgsz,
            }
        )

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t_start
            print(f'  {i + 1}/{len(dataset)} images ({elapsed:.1f}s)', flush=True)

    elapsed = time.perf_counter() - t_start
    ms_per_img = elapsed / max(len(results), 1) * 1000
    print(f'  Done: {len(results)} images in {elapsed:.1f}s ({ms_per_img:.1f} ms/img)')
    print(f'  Predictions: {n_pred_total}, GT: {n_gt_total}')

    # Metrics (reuse eval_pannuke.py streaming)
    from raycasted.scripts.eval_pannuke import compute_metrics_streaming, _diagnose_recall

    print('Computing metrics...')
    metrics = compute_metrics_streaming(results, num_classes=nc)

    ap = metrics['ap']
    ap50 = ap.get(0.5, {}).get('AP', 0.0)
    ap50_95 = np.mean([ap[t]['AP'] for t in sorted(ap.keys())])
    f12 = metrics['centroid']

    print('\n' + '=' * 60)
    print(f'Evaluation Results ({"TensorRT" if use_trt else "ONNX Runtime"})')
    print('=' * 60)
    print(f'{"Metric":<25} {"Value":>12}')
    print('-' * 37)
    print(f'{"AJI":<25} {metrics["aji"]:>12.4f}')
    print(f'{"AP@0.5":<25} {ap50:>12.4f}')
    print(f'{"AP@0.5:0.05:0.95":<25} {ap50_95:>12.4f}')
    print(f'{"bPQ":<25} {metrics["bpq"]:>12.4f}')
    print(f'{"mPQ":<25} {metrics["mpq"]:>12.4f}')
    print(f'{"F1 (r=12)":<25} {f12["f1"]:>12.4f}')
    print(f'{"Precision":<25} {f12["precision"]:>12.4f}')
    print(f'{"Recall":<25} {f12["recall"]:>12.4f}')
    print(f'{"Inference (ms/img)":<25} {ms_per_img:>12.1f}')
    print('=' * 37)

    _diagnose_recall(results, num_classes=nc)


if __name__ == '__main__':
    main()
