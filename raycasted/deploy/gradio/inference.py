"""RayCastED — Gradio Demo Inference Wrapper.

Loads a trained RayCastED checkpoint and runs single-image polygon
detection. Reuses existing postprocessing and annotation code.

Usage:
    from raycasted.deploy.gradio.inference import RayCastDemoInference

    inf = RayCastDemoInference('runs/detect/train/weights/best.pt')
    annotated_img, stats = inf.predict(image_array, conf_threshold=0.25)
"""

import numpy as np
import torch

from raycasted.export.postprocess import postprocess_raw_output
from raycasted.model.annotate import RayCastAnnotator
from raycasted.model.register import register_raycast_head


class RayCastDemoInference:
    """Load a trained RayCastED model and run polygon detection on images.

    Args:
        weights_path: Path to a .pt checkpoint saved by RayCastTrainer.
        device: 'auto', 'cpu', or 'cuda'. Auto picks cuda if available.
    """

    def __init__(self, weights_path: str, device: str = 'auto'):
        register_raycast_head()

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = torch.load(weights_path, map_location=self.device, weights_only=False)
        if hasattr(self.model, 'float'):
            self.model = self.model.float()
        self.model.eval()

        # Read training metadata from checkpoint
        self.training_args = getattr(self.model, 'training_args', {})
        self.imgsz = self.training_args.get('imgsz', 640)
        self.nc = self.training_args.get('nc', 1)
        self.strides = self.training_args.get('strides', [8, 16, 32])
        self.names = getattr(self.model, 'names', {})

        # Build head wrapper for raw logit extraction
        self._wrapper = _ForwardWrapper(self.model)

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        show_rays: bool = False,
    ) -> tuple[np.ndarray, str]:
        """Run detection on a single image.

        Args:
            image: RGB image as [H, W, 3] uint8 numpy array.
            conf_threshold: Confidence threshold for detection filtering.
            show_rays: Whether to draw ray lines on the output.

        Returns:
            Tuple of (annotated_image, stats_text).
            annotated_image: [H, W, 3] uint8 with polygons drawn.
            stats_text: Human-readable detection summary.
        """
        if image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8), 'No image provided.'

        # 1. Preprocess: letterbox resize to imgsz
        letterboxed, scale, pad = _letterbox(image, self.imgsz)
        tensor = torch.from_numpy(letterboxed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        # 2. Forward pass → raw logits [1, N_anchors, nc + 34]
        raw_output = self._wrapper(tensor).cpu().numpy()

        # 3. Postprocess: activations + decode + filter + dedup
        detections = postprocess_raw_output(
            raw_output,
            strides=self.strides,
            imgsz=self.imgsz,
            conf_threshold=conf_threshold,
            dedup_radius_px=min(5.0, self.imgsz * 0.008),
        )
        det = detections[0]  # [N, 36] — single image

        # 4. Scale detections from letterboxed space back to original image space
        if det.shape[0] > 0:
            det[:, 0] = (det[:, 0] - pad[0]) / scale  # cx
            det[:, 1] = (det[:, 1] - pad[1]) / scale  # cy
            det[:, 2:34] = det[:, 2:34] / scale  # rays

        # 5. Draw polygons on original image
        annotated = image.copy()
        annotator = RayCastAnnotator(annotated, show_rays=show_rays)
        annotator.draw_polygons(det)
        annotated = annotator.result()

        # 6. Build stats text
        n_det = det.shape[0]
        if n_det == 0:
            stats = f'No detections (conf > {conf_threshold:.2f})'
        else:
            class_counts = {}
            for row in det:
                cls_idx = int(row[35])
                cls_name = self.names.get(cls_idx, f'class_{cls_idx}')
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            parts = [f'{name}: {count}' for name, count in sorted(class_counts.items())]
            stats = f'Detections: {n_det} cells ({", ".join(parts)})'

        return annotated, stats


class _ForwardWrapper(torch.nn.Module):
    """Wraps the full model to extract raw head logits for postprocessing.

    Runs backbone+neck → forward_head only (no _inference, no postprocess).
    Returns [B, N_anchors, nc + 34] raw logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.head = model.model[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: [B, 3, H, W] preprocessed image tensor.

        Returns:
            [B, N_anchors, nc + 34] raw logits.
        """
        # Backbone + neck
        y = x
        for m in self.model.model[:-1]:
            y = m(y)
        if not isinstance(y, (list, tuple)):
            y = [y]

        # Head forward (raw logits, no activations)
        head = self.head
        preds = head.forward_head(y, box_head=head.cv2, cls_head=head.cv3)

        boxes = preds['boxes']  # [B, 34, N_anchors]
        scores = preds['scores']  # [B, nc, N_anchors]
        return torch.cat([boxes, scores], dim=1).permute(0, 2, 1)  # [B, N, 34+nc]


def _letterbox(
    image: np.ndarray,
    target_size: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize and pad image to target_size x target_size (letterbox).

    Args:
        image: [H, W, 3] uint8 image.
        target_size: Output square size.
        color: Padding color.

    Returns:
        Tuple of (letterboxed_image, scale_factor, (pad_x, pad_y)).
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    letterboxed = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    letterboxed[int(pad_y) : int(pad_y) + new_h, int(pad_x) : int(pad_x) + new_w] = resized

    return letterboxed, scale, (pad_x, pad_y)
