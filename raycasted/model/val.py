"""RayCastED — Polygon Validation Metrics (Phase 8).

RayCastValidator subclasses DetectionValidator, replacing bounding-box IoU
with GPU-accelerated Polar IoU. Reports both shapely_f1 (primary) and
centroid_f1 (LSP-DETR comparison).

Spec reference: docs/project.md section 15
"""

from pathlib import Path

import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class

from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch

RAYCAST_DIM = 34  # xy(2) + rays(32)


# ---------------------------------------------------------------------------
# RayCastDetMetrics
# ---------------------------------------------------------------------------


class RayCastDetMetrics(DetMetrics):
    """Polygon detection metrics: polar mAP + centroid F1.

    Extends DetMetrics with two additional metric tracks:
      - shapely: mAP computed using Polar IoU (primary, same metric as training loss)
      - centroid: F1 computed using centroid Euclidean distance (LSP-DETR comparison)

    Stats dict keys:
      - tp_shapely: Polar IoU true-positive matrix
      - tp_centroid: centroid distance matching true-positive matrix
      - conf, pred_cls, target_cls, target_img: shared across all tracks
    """

    def __init__(self, names=None):
        super().__init__(names or {})
        self.shapely = Metric()
        self.centroid = Metric()
        self.stats = dict(tp_shapely=[], tp_centroid=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.box = Metric()  # kept for DetMetrics compatibility but unused

    def process(self, save_dir=Path('.'), plot=False, on_plot=None):
        """Compute mAP from shapely and centroid true-positive stats."""
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}

        # Always compute GT counts — needed even when all predictions are filtered
        if len(stats.get('target_cls', [])) > 0:
            self.nt_per_class = np.bincount(stats['target_cls'].astype(int), minlength=len(self.names))
            self.nt_per_image = np.bincount(stats['target_img'].astype(int), minlength=len(self.names))
        else:
            self.nt_per_class = np.zeros(len(self.names), dtype=int)
            self.nt_per_image = np.zeros(len(self.names), dtype=int)

        if len(stats.get('tp_shapely', [])) == 0:
            return stats

        # Shapely polygon mAP
        results_shapely = ap_per_class(
            stats['tp_shapely'],
            stats['conf'],
            stats['pred_cls'],
            stats['target_cls'],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix='Poly',
        )[2:]
        self.shapely.nc = len(self.names)
        self.shapely.update(results_shapely)

        # Centroid F1
        results_centroid = ap_per_class(
            stats['tp_centroid'],
            stats['conf'],
            stats['pred_cls'],
            stats['target_cls'],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix='Centroid',
        )[2:]
        self.centroid.nc = len(self.names)
        self.centroid.update(results_centroid)

        return stats

    @property
    def keys(self):
        """Return metric key names for both shapely and centroid tracks."""
        return [
            'metrics/precision(P)',
            'metrics/recall(P)',
            'metrics/mAP50(P)',
            'metrics/mAP50-95(P)',
            'metrics/precision(C)',
            'metrics/recall(C)',
            'metrics/mAP50(C)',
            'metrics/mAP50-95(C)',
        ]

    def mean_results(self):
        """Return mean results for shapely and centroid tracks."""
        return self.shapely.mean_results() + self.centroid.mean_results()

    def class_result(self, i):
        """Return per-class results for shapely and centroid tracks."""
        return self.shapely.class_result(i) + self.centroid.class_result(i)

    @property
    def maps(self):
        """Return mAP scores per class (shapely track)."""
        return self.shapely.maps

    @property
    def fitness(self):
        """Fitness based on shapely mAP50-95 (primary metric)."""
        return self.shapely.fitness()

    @property
    def results_dict(self):
        """Return dict of all metrics."""
        keys = [*self.keys, 'fitness']
        values = [*self.mean_results(), self.fitness]
        return dict(zip(keys, values))


# ---------------------------------------------------------------------------
# RayCastValidator
# ---------------------------------------------------------------------------


class RayCastValidator(DetectionValidator):
    """Polygon detection validator with GPU Polar IoU and centroid F1.

    Subclasses DetectionValidator, replacing bounding-box IoU with GPU-accelerated
    Polar IoU (same metric used in training loss). Reports both polar mAP (primary)
    and centroid F1 (for LSP-DETR comparability).

    No NMS — the end-to-end model already does top-k selection via
    Hungarian matching. Distance-based dedup is handled by RayCastPredictor.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = 'detect'
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.metrics = RayCastDetMetrics()
        self.centroid_thresholds = [6.0, 8.0, 10.0]  # px, LSP-DETR comparability
        self.n_centroid = len(self.centroid_thresholds)

    def preprocess(self, batch):
        """Move batch to device without /255 — images already normalised by RayCastTileDataset.

        Parent DetectionValidator.preprocess divides by 255, but our dataset
        already returns float32 images in [0, 1]. Double-normalising produces
        near-zero inputs that kill all predictions, causing mAP collapse.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == 'cuda')
        batch['img'] = batch['img'].half() if self.args.half else batch['img'].float()
        return batch

    def init_metrics(self, model):
        """Initialize validation metrics for polygon detection."""
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, 'end2end', False)
        self.seen = 0
        self.jdict = []
        self.metrics = RayCastDetMetrics(names=model.names)
        self.confusion_matrix = None  # skip — incompatible with polygon format

    def finalize_metrics(self, *args, **kwargs):
        """Skip confusion matrix plotting — incompatible with 34-dim polygon data."""

    def postprocess(self, preds):
        """Extract polygon predictions from end-to-end model output (no NMS).

        Args:
            preds: Raw model output. preds[0] is [B, max_det, 36] from head.postprocess.

        Returns:
            list[dict] with keys 'bboxes' [N,34], 'conf' [N], 'cls' [N].
        """
        pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds

        outputs = []
        for i in range(pred_tensor.shape[0]):
            det = pred_tensor[i]  # [max_det, 36]
            det = det[det[:, RAYCAST_DIM] > self.args.conf]
            outputs.append(
                {
                    'bboxes': det[:, :RAYCAST_DIM],  # [N, 34] cx, cy, d_1..d_32
                    'conf': det[:, RAYCAST_DIM],  # [N]
                    'cls': det[:, RAYCAST_DIM + 1],  # [N]
                }
            )
        return outputs

    def _prepare_batch(self, si, batch):
        """Extract per-image GT polygons and denormalise to pixel space.

        Args:
            si: Image index within batch.
            batch: Collated batch dict with 'batch_idx', 'cls', 'bboxes' keys.

        Returns:
            dict with 'cls', 'bboxes' [N_gt, 34] in letterboxed pixel space.
        """
        idx = batch['batch_idx'] == si
        cls = batch['cls'][idx].flatten()
        poly = batch['bboxes'][idx]  # [N_gt, 34] normalised
        imgsz = batch['img'].shape[2:]

        if cls.numel():
            # Denormalise: multiply by letterboxed size (crop_size)
            crop_size = imgsz[0]  # square image
            poly = poly.clone()
            poly[:, 0] *= crop_size  # cx
            poly[:, 1] *= crop_size  # cy
            poly[:, 2:] *= crop_size  # rays

        return {
            'cls': cls,
            'bboxes': poly,
            'ori_shape': batch['ori_shape'][si],
            'imgsz': imgsz,
            'ratio_pad': batch['ratio_pad'][si],
            'im_file': batch['im_file'][si],
        }

    def _prepare_pred(self, pred):
        """Prepare polygon predictions (passthrough, single_cls handling)."""
        if self.args.single_cls:
            pred['cls'] *= 0
        return pred

    def _process_batch(self, preds, batch):
        """Compute GPU Polar IoU and centroid distance matches.

        Uses the same polar IoU as training loss (consistent metric) computed
        entirely on GPU. Only the final TP matrix transfers to CPU.

        Args:
            preds: dict with 'bboxes' [N_pred, 34], 'conf', 'cls'.
            batch: dict with 'bboxes' [N_gt, 34], 'cls' from _prepare_batch.

        Returns:
            dict with 'tp_shapely' and 'tp_centroid' arrays.
        """
        n_pred = preds['cls'].shape[0]
        n_gt = batch['cls'].shape[0]

        if n_gt == 0 or n_pred == 0:
            return {
                'tp_shapely': np.zeros((n_pred, self.niou), dtype=bool),
                'tp_centroid': np.zeros((n_pred, self.n_centroid), dtype=bool),
            }

        # Extract rays from polygon tensors (keep on GPU)
        pred_rays = preds['bboxes'][:, 2:]  # [N_pred, 32]
        gt_rays = batch['bboxes'][:, 2:]  # [N_gt, 32]

        # Expand to pairwise shape for polar IoU on GPU
        pred_exp = pred_rays[:, None, :].expand(n_pred, n_gt, 32)
        gt_exp = gt_rays[None, :, :].expand(n_pred, n_gt, 32)
        iou_matrix = polar_iou_pairwise_flat_torch(pred_exp, gt_exp)  # [N_pred, N_gt]

        # Polar IoU true-positive matching (same metric as training loss)
        # match_predictions expects iou shape [N_gt, N_pred] (parent convention)
        tp_shapely = (
            self.match_predictions(preds['cls'], batch['cls'], iou_matrix.T).cpu().numpy()
        )

        # Centroid distance matching (CPU — small matrices, fast)
        tp_centroid = self._compute_centroid_matches(
            preds['bboxes'][:, :2].cpu().numpy(),
            batch['bboxes'][:, :2].cpu().numpy(),
            preds['cls'].cpu(),
            batch['cls'].cpu(),
        )

        return {'tp_shapely': tp_shapely, 'tp_centroid': tp_centroid}

    def _compute_centroid_matches(self, pred_centroids, gt_centroids, pred_cls, gt_cls):
        """Match predictions to GT by centroid Euclidean distance.

        For each distance threshold, creates a binary match matrix and applies
        the same greedy matching as match_predictions.

        Args:
            pred_centroids: [N, 2] predicted centroid (cx, cy).
            gt_centroids: [M, 2] ground truth centroid (cx, cy).
            pred_cls: [N] predicted class indices.
            gt_cls: [M] ground truth class indices.

        Returns:
            np.ndarray [N, n_centroid_thresholds] bool — true-positive matrix.
        """
        n_pred = len(pred_centroids)
        n_thresh = len(self.centroid_thresholds)
        correct = np.zeros((n_pred, n_thresh), dtype=bool)

        # Pairwise Euclidean distance matrix
        dists = np.sqrt(((pred_centroids[:, None, :] - gt_centroids[None, :, :]) ** 2).sum(axis=2))

        # Class matching — transpose to match dists shape [N_pred, N_gt]
        correct_class = (gt_cls[:, None] == pred_cls).T  # [N_pred, N_gt]
        if isinstance(correct_class, torch.Tensor):
            correct_class = correct_class.cpu().numpy()

        for t_idx, threshold in enumerate(self.centroid_thresholds):
            match_matrix = (dists <= threshold) * correct_class  # [N_pred, N_gt]

            # Greedy matching: sort by distance ascending, unique per row and column
            matches = np.nonzero(match_matrix)
            matches = np.array(matches).T  # [K, 2] (pred_idx, gt_idx)
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[dists[matches[:, 0], matches[:, 1]].argsort()]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                correct[matches[:, 0].astype(int), t_idx] = True

        return correct

    def update_metrics(self, preds, batch):
        """Update metrics with polygon predictions and ground truth.

        Overridden to pass tp_shapely/tp_centroid and skip confusion matrix.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch['cls'].cpu().numpy()
            no_pred = predn['cls'].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    'target_cls': cls,
                    'target_img': np.unique(cls),
                    'conf': np.zeros(0) if no_pred else predn['conf'].cpu().numpy(),
                    'pred_cls': np.zeros(0) if no_pred else predn['cls'].cpu().numpy(),
                }
            )

    def get_stats(self):
        """Compute and return validation metrics."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def plot_predictions(self, batch, preds, ni):
        """Skip standard bbox prediction plotting — incompatible with 34-dim polygon data."""

    def plot_val_samples(self, batch, ni):
        """Skip standard bbox val sample plotting — incompatible with 34-dim polygon data."""