"""NuLite-DETR — ONNX export for deployment (Jetson TensorRT).

Exports the query-based NuLite-DETR model to ONNX. The graph is the raw
decoder output only — no activation decisions, no NMS, no confidence
threshold. The NMS-free inference contract is preserved: the retained/empty
decision stays the raw ``nc+1`` argmax computed on the host from the exported
logits, exactly as ``raycasted.evals.eval_lsp`` does.

Outputs (dynamic batch axis, fixed 256x256 input):
  - dec_polygons : [B, nq, 2 + n_rays]  normalized polygon (sigmoid xy, rays in [0,1])
  - dec_logits   : [B, nq, nc + 1]      raw class logits, last column = no-object

These are exactly the tensors eval_lsp.py consumes (out5[0], out5[1]), so the
host-side decode (argmax over nc+1, drop no-object, polys * crop) matches the
literature-facing evaluator bit-for-bit.

Usage:
    uv run python -m raycasted.export.export_nulite_detr \
        --weights output/v2/train52_fastvit-t12/weights/last.pt \
        --output-dir output/v2/train52_fastvit-t12/deploy \
        --validate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import raycasted.model.register  # noqa: F401  (registers custom modules for unpickling)
from raycasted.evals.eval_pannuke import load_model


class NuLiteDetrExportWrapper(nn.Module):
    """Return the exact tensors eval_lsp.py consumes from the raw model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Return (dec_polygons (B,nq,66) normalized, dec_logits (B,nq,nc+1))."""
        y_detect, out5 = self.model.predict(x)
        del y_detect
        dec_polys = out5[0].squeeze(0)  # (B, nq, raycast_dim), normalized
        dec_logits = out5[1].squeeze(0)  # (B, nq, nc+1), raw logits
        return dec_polys, dec_logits


def load_checkpoint(weights_path: str):
    """Load the eval-mode NuLite-DETR checkpoint and build its meta dict."""
    model = load_model(weights_path, torch.device('cpu'))
    tr = getattr(model, 'training_args', {})
    meta = {
        'arch': 'nulite-detr',
        'crop_size': int(tr.get('crop_size', 256)),
        'imgsz': int(tr.get('crop_size', 256)),
        'nc': int(tr.get('nc', getattr(model, 'nc', 5))),
        'n_rays': int(tr.get('n_rays', 64)),
        'raycast_dim': 2 + int(tr.get('n_rays', 64)),
        'no_object': bool(getattr(model, 'no_object', False)),
        'oracle_kind': getattr(model, 'oracle_kind', None),
        'input_range': '0..1 (CHW float32, as produced by RayCastTileDataset augment=False)',
        'decode': (
            'keep = argmax(logits, -1) != nc (last col); label = argmax over first nc cols; '
            'score = 1 - sigmoid(logit[:, nc]); pixels = poly * crop_size. '
            'No NMS, no confidence threshold.'
        ),
    }
    return model, meta


def export_nulite_detr_onnx(
    weights_path: str,
    output_path: str,
    imgsz: int = 256,
    opset: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> dict:
    """Export the NuLite-DETR eval graph to ONNX and return export metadata."""
    model, meta = load_checkpoint(weights_path)
    wrapper = NuLiteDetrExportWrapper(model).eval()

    dummy = torch.rand(1, 3, imgsz, imgsz)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch'},
            'dec_polygons': {0: 'batch'},
            'dec_logits': {0: 'batch'},
        }

    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        opset_version=opset,
        input_names=['images'],
        output_names=['dec_polygons', 'dec_logits'],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    if simplify:
        try:
            import onnxsim

            simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(simplified, output_path)
            else:
                print('onnxsim returned check=False; keeping the unsimplified graph')
        except ImportError:
            print('onnxsim not installed; skipping simplification')

    meta['opset'] = opset
    meta['dynamic_batch'] = dynamic_batch
    meta['weights'] = str(weights_path)
    meta['onnx_sha256'] = _sha256(output_path)
    return meta


def validate_onnx(
    onnx_path: str,
    weights_path: str,
    imgsz: int = 256,
    batches=(1, 3),
    atol: float = 2e-4,
    rtol: float = 1e-3,
) -> dict:
    """Validate the exported graph against the PyTorch eval path (CPU)."""
    model, _ = load_checkpoint(weights_path)
    wrapper = NuLiteDetrExportWrapper(model).eval()

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    rng = np.random.default_rng(0)
    results = []
    all_passed = True
    for b in batches:
        x = torch.from_numpy(rng.random((b, 3, imgsz, imgsz), dtype=np.float32))
        with torch.no_grad():
            pt = wrapper(x)
        out = session.run(None, {'images': x.numpy()})
        row = {'batch': b}
        for name, pt_val, ort_val in zip(['dec_polygons', 'dec_logits'], pt, out):
            max_d = float(np.abs(pt_val.numpy() - ort_val).max())
            row[name] = {'shape': list(ort_val.shape), 'max_diff': max_d}
            all_passed &= bool(np.allclose(pt_val.numpy(), ort_val, atol=atol, rtol=rtol))
        results.append(row)
    return {'passed': all_passed, 'details': results}


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    """CLI: export a NuLite-DETR checkpoint to ONNX + meta.json (+ validate)."""
    ap = argparse.ArgumentParser(description='Export NuLite-DETR checkpoint to ONNX for TensorRT')
    ap.add_argument('--weights', required=True, help='Path to .pt checkpoint (e.g. .../weights/last.pt)')
    ap.add_argument('--output-dir', default=None, help='Output dir (default: <weights dir>/../deploy)')
    ap.add_argument('--name', default=None, help='Output basename (default: checkpoint dir name)')
    ap.add_argument('--imgsz', type=int, default=256, help='Input size (default 256, must match training crop)')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--static-batch', action='store_true', help='Export with fixed batch=1')
    ap.add_argument('--no-simplify', action='store_true')
    ap.add_argument('--validate', action='store_true', help='Compare ORT vs PyTorch on CPU')
    args = ap.parse_args()

    weights = Path(args.weights)
    out_dir = Path(args.output_dir) if args.output_dir else weights.parent.parent / 'deploy'
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or weights.parent.parent.name
    onnx_path = out_dir / f'{name}.onnx'

    meta = export_nulite_detr_onnx(
        str(weights),
        str(onnx_path),
        args.imgsz,
        args.opset,
        dynamic_batch=not args.static_batch,
        simplify=not args.no_simplify,
    )
    meta_path = onnx_path.with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Exported: {onnx_path}')
    print(f'Meta:     {meta_path}')

    if args.validate:
        result = validate_onnx(str(onnx_path), str(weights), args.imgsz)
        print(f'Validation: {"PASSED" if result["passed"] else "FAILED"}')
        for row in result['details']:
            print(f"  batch={row['batch']}: " + ', '.join(
                f"{k}: shape={v['shape']} max_diff={v['max_diff']:.2e}" for k, v in row.items() if isinstance(v, dict)
            ))
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
