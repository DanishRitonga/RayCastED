"""RayCastED — Jetson Post-Processing.

NumPy-based post-processing for raw ONNX model output. Runs on the
Jetson host (not in the ONNX graph). Applies activations, decodes
xy coordinates, denormalises rays, and filters by confidence.

No NMS, no distance-based dedup, no clustering. The model produces
the correct set of predictions directly from the o2o head.
"""

import numpy as np


def postprocess_raw_output(
    raw_boxes: np.ndarray,
    raw_binary: np.ndarray | None,
    raw_class: np.ndarray,
    strides: list[float],
    imgsz: int = 256,
    conf_threshold: float = 0.20,
    binary_threshold: float = 0.01,
    n_rays: int = 64,
    max_det: int = 100,
) -> np.ndarray:
    """Post-process raw ONNX head output to polygon detections.

    Args:
        raw_boxes: [N_anchors, 2+n_rays] polygon logits.
        raw_binary: [N_anchors, 1] fg/bg logits (hierarchical), or None.
        raw_class: [N_anchors, nc] class logits.
        strides: Feature map strides (e.g. [4, 8, 16]).
        imgsz: Input image size (square).
        conf_threshold: Confidence threshold for filtering.
        binary_threshold: Hard gate threshold for binary head.
        n_rays: Number of radial rays.
        max_det: Maximum detections per image (top-K by confidence).

    Returns:
        [N_det, 4+n_rays] array: [cx, cy, d_1..d_n, score, cls_idx].
    """
    raycast_dim = 2 + n_rays

    if raw_boxes.ndim == 3:
        raw_boxes = raw_boxes[0]
        raw_class = raw_class[0]
        if raw_binary is not None:
            raw_binary = raw_binary[0]

    if raw_binary is not None and raw_binary.ndim == 2:
        raw_binary = raw_binary.squeeze(-1)

    xy_offset = 1.0 / (1.0 + np.exp(-raw_boxes[:, :2]))
    rays = np.log1p(np.exp(raw_boxes[:, 2:]))

    if raw_binary is not None:
        binary_scores = 1.0 / (1.0 + np.exp(-raw_binary))
        cls_scores = 1.0 / (1.0 + np.exp(-raw_class))
        combined = binary_scores * cls_scores.max(axis=1, keepdims=True)
        cls_idx = cls_scores.argmax(axis=1)
        max_scores = combined
        hard_gate = binary_scores < binary_threshold
        max_scores[hard_gate] = 0.0
    else:
        cls_scores = 1.0 / (1.0 + np.exp(-raw_class))
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

    indices = np.where(mask)[0]
    if n_det > max_det:
        topk = np.argsort(-max_scores[indices])[:max_det]
        indices = indices[topk]
        n_det = max_det

    det = np.zeros((n_det, raycast_dim + 2), dtype=np.float32)
    det[:, 0] = cx[indices]
    det[:, 1] = cy[indices]
    det[:, 2:2 + n_rays] = rays_px[indices]
    det[:, 2:2 + n_rays] = rays_px[mask]
    det[:, raycast_dim] = max_scores[indices]
    det[:, raycast_dim + 1] = cls_idx[indices]
    return det


def _build_anchor_grid(strides: list[float], imgsz: int):
    all_ax, all_ay, all_s = [], [], []
    for stride in strides:
        feat_size = int(imgsz / stride)
        gx, gy = np.meshgrid(np.arange(feat_size), np.arange(feat_size))
        all_ax.append((gx + 0.5).flatten().astype(np.float32))
        all_ay.append((gy + 0.5).flatten().astype(np.float32))
        all_s.append(np.full(feat_size * feat_size, stride, dtype=np.float32))
    return np.concatenate(all_ax), np.concatenate(all_ay), np.concatenate(all_s)


def postprocess_batch(
    outputs: dict[str, np.ndarray] | list[np.ndarray],
    strides: list[float],
    imgsz: int = 256,
    conf_threshold: float = 0.20,
    binary_threshold: float = 0.01,
    n_rays: int = 64,
) -> list[np.ndarray]:
    """Post-process a batch of ONNX outputs.

    Args:
        outputs: Dict with 'boxes', 'binary', 'class' keys (hierarchical)
                 or 'boxes', 'scores' keys (standard).
        strides: Feature map strides.
        imgsz: Input image size.
        conf_threshold: Confidence threshold.
        binary_threshold: Binary head hard gate.
        n_rays: Number of radial rays.

    Returns:
        list of [N_det, 4+n_rays] arrays, one per image.
    """
    if isinstance(outputs, dict):
        boxes = outputs['boxes']
        binary = outputs.get('binary')
        scores = outputs.get('scores')
        class_scores = scores if scores is not None else outputs.get('class', outputs['scores'])
        if binary is not None and binary.ndim == 3:
            binary = binary.squeeze(1)
    elif isinstance(outputs, (list, tuple)):
        if len(outputs) == 3:
            boxes, binary, class_scores = outputs
        else:
            boxes, class_scores = outputs
            binary = None
    else:
        raise ValueError(f'Unexpected output format: {type(outputs)}')

    batch_size = boxes.shape[0] if boxes.ndim == 4 else 1
    results = []
    for b in range(batch_size):
        b_boxes = boxes[b] if boxes.ndim == 4 else boxes
        b_binary = binary[b] if binary is not None and binary.ndim >= 2 else binary
        b_class = class_scores[b] if class_scores.ndim >= 3 else class_scores
        det = postprocess_raw_output(
            b_boxes, b_binary, b_class, strides, imgsz, conf_threshold, binary_threshold, n_rays
        )
        results.append(det)
    return results
