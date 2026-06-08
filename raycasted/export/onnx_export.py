"""RayCastED — ONNX Export Utilities.

Exports the RayCastED model to ONNX format for TensorRT deployment on
NVIDIA Jetson. The ONNX graph contains raw head logits only — no
activations, decoding, or post-processing. All post-processing runs
on the Jetson host in Python/NumPy.

Exports three output tensors:
  - boxes:    [B, raycast_dim, N] polygon logits (xy + rays)
  - binary:   [B, 1, N]           fg/bg logits (hierarchical_cls)
  - class:    [B, nc, N]          class logits

For models without hierarchical_cls, binary and class are merged into
a single 'scores' output.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from raycasted.data.etl.utils import constants as _const


class _ExportWrapper(nn.Module):
    """Wraps the RayCastED model to export raw o2o head logits.

    Runs backbone+neck, then the o2o head's forward_head(). Exports
    raw logits without activations, decoding, or post-processing.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        inner = model.model if hasattr(model, 'model') else model
        self.head = inner[-1]
        self._inner = inner

    def forward(self, x):
        y = x
        for i, m in enumerate(self._inner[:-1]):
            y = m(y)
        if not isinstance(y, (list, tuple)):
            y = [y]

        head = self.head
        if self._hierarchical:
            preds = head.forward_head(
                y,
                box_head=head.one2one_cv2,
                cls_head_binary=head.one2one_cv3_binary,
                cls_head_class=head.one2one_cv3_class,
            )
            return (
                preds['boxes'],          # [B, 66, N]
                preds['binary_scores'],  # [B, 1, N]
                preds['class_scores'],   # [B, nc, N]
            )
        else:
            preds = head.forward_head(
                y, box_head=head.one2one_cv2, cls_head=head.one2one_cv3
            )
            return preds['boxes'], preds['scores']


def export_raycast_onnx(
    weights_path: str,
    output_path: str,
    imgsz: int = 256,
    opset: int = 17,
    simplify: bool = True,
    dynamic_batch: bool = False,
) -> str:
    """Export RayCastED model to ONNX.

    Args:
        weights_path: Path to .pt checkpoint.
        output_path: Path to write .onnx file.
        imgsz: Input image size (square). Default 256.
        opset: ONNX opset version.
        simplify: Run onnx-simplifier to fold constants.
        dynamic_batch: Export with dynamic batch dimension.

    Returns:
        Path to the exported .onnx file.
    """
    from raycasted.model.register import register_raycast_head

    register_raycast_head()

    model = torch.load(weights_path, map_location='cpu', weights_only=False)
    if isinstance(model, dict):
        model = model.get('model') or model.get('ema') or model
    if hasattr(model, 'float'):
        model = model.float()
    model.eval()

    from raycasted.export.dcn_strip import strip_dcn_from_model

    inner = model.model if hasattr(model, 'model') else model
    strip_dcn_from_model(inner)

    wrapper = _ExportWrapper(model)

    dummy = torch.randn(1, 3, imgsz, imgsz)
    head = inner[-1]
    hierarchical = getattr(head, 'hierarchical_cls', False)

    output_names = ['boxes', 'binary', 'class'] if hierarchical else ['boxes', 'scores']

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'images': {0: 'batch'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch'}

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=opset,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    if simplify:
        try:
            import onnxsim
            onnx_model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model_simplified, output_path)
        except ImportError:
            pass

    strides = head.stride.tolist() if hasattr(head, 'stride') else [4, 8, 16]
    meta = {
        'crop_size': imgsz,
        'imgsz': imgsz,
        'nc': head.nc,
        'n_rays': head.n_rays,
        'ray_cos': _const.RAY_COS.tolist(),
        'ray_sin': _const.RAY_SIN.tolist(),
        'strides': strides,
        'conf_threshold': 0.20,
        'hierarchical_cls': hierarchical,
        'binary_threshold': getattr(head, 'hierarchical_binary_threshold', 0.01),
    }
    meta_path = Path(output_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return output_path


def validate_onnx(
    onnx_path: str,
    weights_path: str,
    imgsz: int = 256,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict:
    """Validate ONNX export against PyTorch model."""
    from raycasted.model.register import register_raycast_head

    register_raycast_head()

    model = torch.load(weights_path, map_location='cpu', weights_only=False)
    if isinstance(model, dict):
        model = model.get('model') or model.get('ema') or model
    if hasattr(model, 'float'):
        model = model.float()
    model.eval()

    from raycasted.export.dcn_strip import strip_dcn_from_model

    inner = model.model if hasattr(model, 'model') else model
    strip_dcn_from_model(inner)

    wrapper = _ExportWrapper(model)
    head = model.model[-1]
    hierarchical = getattr(head, 'hierarchical_cls', False)

    dummy = torch.randn(1, 3, imgsz, imgsz)

    with torch.no_grad():
        pt_output = wrapper(dummy)

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {'images': dummy.numpy()})

    if not isinstance(pt_output, tuple):
        pt_output = (pt_output,)

    max_diffs = []
    all_passed = True
    names = ['boxes', 'binary', 'class'] if hierarchical else ['boxes', 'scores']
    for i, name in enumerate(names):
        pt_val = pt_output[i].numpy()
        onnx_val = onnx_output[i]
        max_d = float(np.abs(pt_val - onnx_val).max())
        mean_d = float(np.abs(pt_val - onnx_val).mean())
        ok = bool(np.allclose(pt_val, onnx_val, atol=atol, rtol=rtol))
        max_diffs.append({'name': name, 'max_diff': max_d, 'mean_diff': mean_d, 'passed': ok})
        all_passed = all_passed and ok

    return {
        'passed': all_passed,
        'shape_match': all(
            pt_output[i].shape == onnx_output[i].shape for i in range(len(names))
        ),
        'details': max_diffs,
    }
