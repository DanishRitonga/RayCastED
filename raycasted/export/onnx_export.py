"""RayCastED — ONNX Export Utilities (Phase 9).

Exports the RayCastED model to ONNX format for TensorRT deployment on
NVIDIA Jetson. The ONNX graph contains raw head logits only — no
activations, decoding, or post-processing. All post-processing runs
on the Jetson host in Python/NumPy.

Spec reference: docs/project.md section 16
"""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from raycasted.data.etl.utils.constants import RAY_COS, RAY_SIN

METADATA_KEYS = [
    'crop_size',
    'imgsz',
    'nc',
    'n_rays',
    'ray_cos',
    'ray_sin',
    'strides',
    'conf_threshold',
    'dedup_radius_px',
]


class _RawHeadWrapper(nn.Module):
    """Wraps the full YOLO model to export raw head logits.

    Runs backbone+neck, then forward_head() only. No activations,
    no _inference(), no postprocess(). Returns [B, N_anchors, nc + 34].

    Args:
        model: The full YOLO DetectionModel.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        head = model.model[-1]
        self.head = head

    def forward(self, x):
        """Run backbone+neck, then raw head output.

        Args:
            x: [B, 3, H, W] input image tensor.

        Returns:
            [B, N_anchors, nc + 34] raw head logits.
        """
        # Run through all layers except the head
        y = x
        for i, m in enumerate(self.model.model[:-1]):
            y = m(y)

        # y should be a list of feature maps for the head
        if not isinstance(y, (list, tuple)):
            y = [y]

        # Run forward_head only — raw logits, no activations
        head = self.head
        preds = head.forward_head(y, box_head=head.cv2, cls_head=head.cv3)

        boxes = preds['boxes']  # [B, 34, N_anchors]
        scores = preds['scores']  # [B, nc, N_anchors]
        return torch.cat([boxes, scores], dim=1).permute(0, 2, 1)  # [B, N, 34+nc]


def export_polygon_yolo_onnx(
    weights_path: str,
    output_path: str,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic_batch: bool = False,
) -> str:
    """Export RayCastED model to ONNX format.

    The ONNX graph contains raw head logits only. Post-processing
    (sigmoid, softplus, xy decoding, ray denormalisation, thresholding,
    dedup) runs on the deployment target in Python/NumPy.

    Args:
        weights_path: Path to .pt checkpoint.
        output_path: Path to write .onnx file.
        imgsz: Input image size (square). Default 640.
        opset: ONNX opset version. 17 covers GroupNorm, Softplus, SiLU.
        simplify: Run onnx-simplifier to fold constants. Default True.
        dynamic_batch: Export with dynamic batch dimension. Default False.

    Returns:
        Path to the exported .onnx file.
    """
    from raycasted.model.register import register_raycast_head

    register_raycast_head()

    model = torch.load(weights_path, map_location='cpu', weights_only=False)
    if hasattr(model, 'float'):
        model = model.float()
    model.eval()

    wrapper = _RawHeadWrapper(model)

    dummy = torch.randn(1, 3, imgsz, imgsz)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'images': {0: 'batch'}, 'output': {0: 'batch'}}

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=opset,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Optionally simplify
    if simplify:
        try:
            import onnxsim

            onnx_model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model_simplified, output_path)
        except ImportError:
            pass  # onnxsim not available, skip simplification

    # Write metadata sidecar
    head = model.model[-1]
    strides = head.stride.tolist() if hasattr(head, 'stride') else [8, 16, 32]
    meta = {
        'crop_size': imgsz,
        'imgsz': imgsz,
        'nc': head.nc,
        'n_rays': 32,
        'ray_cos': RAY_COS.tolist(),
        'ray_sin': RAY_SIN.tolist(),
        'strides': strides,
        'conf_threshold': 0.25,
        'dedup_radius_px': 5,
    }
    meta_path = Path(output_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return output_path


def validate_onnx(
    onnx_path: str,
    weights_path: str,
    imgsz: int = 640,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict:
    """Validate ONNX export against PyTorch model.

    Runs both models on the same input and checks numerical agreement.

    Args:
        onnx_path: Path to .onnx file.
        weights_path: Path to .pt checkpoint.
        imgsz: Input image size.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        dict with 'max_diff', 'mean_diff', 'passed' keys.
    """
    from raycasted.model.register import register_raycast_head

    register_raycast_head()

    model = torch.load(weights_path, map_location='cpu', weights_only=False)
    if hasattr(model, 'float'):
        model = model.float()
    model.eval()

    wrapper = _RawHeadWrapper(model)

    dummy = torch.randn(1, 3, imgsz, imgsz)

    with torch.no_grad():
        pt_output = wrapper(dummy).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {'images': dummy.numpy()})[0]

    max_diff = np.abs(pt_output - onnx_output).max()
    mean_diff = np.abs(pt_output - onnx_output).mean()
    passed = bool(np.allclose(pt_output, onnx_output, atol=atol, rtol=rtol))

    return {
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'passed': passed,
        'shape_match': pt_output.shape == onnx_output.shape,
        'pt_shape': list(pt_output.shape),
        'onnx_shape': list(onnx_output.shape),
    }
