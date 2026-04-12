"""RayCastED — RayCast Prediction Pipeline (Phase 7).

RayCastPredictor subclasses DetectionPredictor, replacing NMS with
distance-based dedup and polygon-aware result construction.

Spec reference: docs/project.md section 14
"""

import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class RayCastPredictor(DetectionPredictor):
    """Polygon detection predictor with distance-based dedup.

    Replaces standard NMS with centroid-distance deduplication,
    appropriate for raycast polygon predictions where bounding-box
    IoU is not meaningful. Assumes end-to-end (NMS-free) inference:
    head.postprocess() already did top-k selection.
    """

    def setup_model(self, model, verbose=True):
        """Set up model and derive raycast_dim from head."""
        super().setup_model(model, verbose)
        head = self.model.model[-1] if hasattr(self.model, 'model') else self.model
        self.raycast_dim = getattr(head, 'raycast_dim', 34)

    def postprocess(self, preds, img, orig_imgs):
        """Post-process polygon predictions into Results objects.

        Args:
            preds: Either [y_tensor, raw_dict] (non-export) or y_tensor (export).
                y_tensor: [B, max_det, raycast_dim+2] from head.postprocess (end2end).
            img: [B, C, H, W] preprocessed image tensor.
            orig_imgs: list of original images (H, W, 3).

        Returns:
            list[Results] with polygon data in result.polygons [N, raycast_dim+2].
        """
        pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        imgsz = img.shape[2]  # letterboxed size (square)
        dedup_radius = min(5.0, imgsz * 0.008)

        results = []
        for i in range(pred_tensor.shape[0]):
            det = pred_tensor[i]  # [max_det, raycast_dim+2]
            det = det[det[:, self.raycast_dim] > self.args.conf]

            if det.shape[0] == 0:
                r = Results(orig_imgs[i], path=self.batch[0][i], names=self.model.names)
                r.polygons = np.zeros((0, self.raycast_dim + 2), dtype=np.float32)
                results.append(r)
                continue

            # Scale from letterboxed space to original image space
            poly = det[:, :self.raycast_dim].clone()
            poly = self._scale_polygons(poly, img.shape[2:], orig_imgs[i].shape[:2])
            det = torch.cat([poly, det[:, self.raycast_dim:]], dim=1)

            # Distance-based dedup on centroids
            keep = self._dedup_by_distance(det[:, :2], det[:, self.raycast_dim], dedup_radius)
            det = det[keep]

            # Class filter
            if self.args.classes is not None:
                cls_mask = torch.tensor(
                    [int(c.item()) in self.args.classes for c in det[:, self.raycast_dim + 1]],
                    device=det.device,
                )
                det = det[cls_mask]

            r = Results(orig_imgs[i], path=self.batch[0][i], names=self.model.names)
            r.polygons = det.cpu().numpy()  # [N_det, raycast_dim+2]
            results.append(r)

        return results

    @staticmethod
    def _scale_polygons(poly, img1_shape, img0_shape):
        """Scale polygon predictions from letterboxed to original image space.

        Args:
            poly: [N, raycast_dim] — [cx, cy, d_1..d_n] in letterboxed pixel space.
            img1_shape: (H, W) of the letterboxed image.
            img0_shape: (H, W) of the original image.

        Returns:
            poly: [N, raycast_dim] scaled to original image space.
        """
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
        pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
        poly[:, 0] = (poly[:, 0] - pad_x) / gain  # cx
        poly[:, 1] = (poly[:, 1] - pad_y) / gain  # cy
        poly[:, 2:] = poly[:, 2:] / gain  # rays (isotropic distances)
        return poly

    @staticmethod
    def _dedup_by_distance(centroids, scores, radius_px):
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
            return torch.ones(n, dtype=torch.bool, device=centroids.device)

        order = scores.argsort(descending=True)
        keep = torch.zeros(n, dtype=torch.bool, device=centroids.device)
        kept_list = []

        for idx in order.tolist():
            if not kept_list:
                keep[idx] = True
                kept_list.append(centroids[idx])
            else:
                kept = torch.stack(kept_list)
                dists = torch.cdist(centroids[idx : idx + 1], kept)[0]
                if dists.min() >= radius_px:
                    keep[idx] = True
                    kept_list.append(centroids[idx])

        return keep
