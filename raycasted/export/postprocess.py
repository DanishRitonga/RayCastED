"""RayCastED — Jetson Post-Processing (Phase 9).

NumPy-based post-processing for raw ONNX model output. Runs on the
Jetson host (not in the ONNX graph). Applies activations, decodes
xy coordinates, denormalises rays, filters by confidence, and
deduplicates detections.

This module is hardware-independent and can be tested on any machine.
"""

import numpy as np

RAYCAST_DIM = 34  # xy(2) + rays(32)


def postprocess_raw_output(
    output: np.ndarray,
    strides: list[float],
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    dedup_radius_px: float = 5.0,
) -> list[np.ndarray]:
    """Post-process raw ONNX head output to polygon detections.

    Applies sigmoid/softplus activations, decodes xy coordinates,
    denormalises rays, filters by confidence, and deduplicates.

    Args:
        output: [B, N_anchors, nc + 34] raw logits from ONNX model.
        strides: Feature map strides [8, 16, 32].
        imgsz: Input image size (square).
        conf_threshold: Confidence threshold for filtering.
        dedup_radius_px: Minimum distance between centroids for dedup.

    Returns:
        list of [N_det, 36] arrays (one per image), where each row is
        [cx, cy, d_1..d_32, score, cls_idx].
    """
    batch_size = output.shape[0]
    results = []

    for i in range(batch_size):
        det = _decode_single(output[i], strides, imgsz)

        # Filter by confidence
        mask = det[:, RAYCAST_DIM] > conf_threshold
        det = det[mask]

        if det.shape[0] == 0:
            results.append(np.zeros((0, RAYCAST_DIM + 2), dtype=np.float32))
            continue

        # Distance-based dedup
        keep = _dedup_by_distance(det[:, :2], det[:, RAYCAST_DIM], dedup_radius_px)
        det = det[keep]

        results.append(det)

    return results


def _decode_single(raw: np.ndarray, strides: list[float], imgsz: int) -> np.ndarray:
    """Decode raw logits for a single image.

    Args:
        raw: [N_anchors, nc + 34] raw logits.
        strides: Feature map strides.
        imgsz: Input image size.

    Returns:
        [N_anchors, 36] decoded detections.
    """
    n_anchors = raw.shape[0]

    poly_logits = raw[:, :RAYCAST_DIM]  # [N, 34]
    cls_logits = raw[:, RAYCAST_DIM:]  # [N, nc]

    # Activations
    xy_offset = 1.0 / (1.0 + np.exp(-poly_logits[:, :2]))  # sigmoid
    rays = np.log1p(np.exp(poly_logits[:, 2:]))  # softplus: log(1 + exp(x))

    cls_scores = 1.0 / (1.0 + np.exp(-cls_logits))  # sigmoid

    # Build anchor grid
    anchor_x, anchor_y, stride_arr = _build_anchor_grid(strides, imgsz)

    # Decode xy: (sigmoid(logit) * 2 - 0.5 + anchor) * stride
    cx = (xy_offset[:, 0] * 2.0 - 0.5 + anchor_x) * stride_arr
    cy = (xy_offset[:, 1] * 2.0 - 0.5 + anchor_y) * stride_arr

    # Denormalise rays: softplus(logit) * imgsz
    rays_px = rays * imgsz

    # Class scores: max score and class index
    max_scores = cls_scores.max(axis=1)
    cls_idx = cls_scores.argmax(axis=1)

    # Assemble: [cx, cy, d_1..d_32, score, cls_idx]
    det = np.zeros((n_anchors, RAYCAST_DIM + 2), dtype=np.float32)
    det[:, 0] = cx
    det[:, 1] = cy
    det[:, 2:34] = rays_px
    det[:, 34] = max_scores
    det[:, 35] = cls_idx

    return det


def _build_anchor_grid(strides: list[float], imgsz: int):
    """Build anchor grid for decoding xy coordinates.

    Args:
        strides: Feature map strides [8, 16, 32].
        imgsz: Input image size (square).

    Returns:
        anchor_x: [N_anchors] x coordinates in grid space.
        anchor_y: [N_anchors] y coordinates in grid space.
        stride_arr: [N_anchors] stride for each anchor.
    """
    all_ax, all_ay, all_s = [], [], []

    for stride in strides:
        feat_size = int(imgsz / stride)
        gx, gy = np.meshgrid(np.arange(feat_size), np.arange(feat_size))
        all_ax.append((gx + 0.5).flatten().astype(np.float32))
        all_ay.append((gy + 0.5).flatten().astype(np.float32))
        all_s.append(np.full(feat_size * feat_size, stride, dtype=np.float32))

    return np.concatenate(all_ax), np.concatenate(all_ay), np.concatenate(all_s)


def _dedup_by_distance(centroids: np.ndarray, scores: np.ndarray, radius_px: float) -> np.ndarray:
    """Greedy dedup: keep highest-confidence detection when centroids overlap.

    Args:
        centroids: [N, 2] pixel-space centroids.
        scores: [N] confidence scores.
        radius_px: minimum distance between centroids.

    Returns:
        keep: boolean mask [N].
    """
    n = centroids.shape[0]
    if n <= 1:
        return np.ones(n, dtype=bool)

    order = np.argsort(-scores)
    keep = np.zeros(n, dtype=bool)
    kept = []

    for idx in order:
        if not kept:
            keep[idx] = True
            kept.append(centroids[idx])
        else:
            dists = np.sqrt(np.sum((centroids[idx] - np.array(kept)) ** 2, axis=1))
            if dists.min() >= radius_px:
                keep[idx] = True
                kept.append(centroids[idx])

    return keep
