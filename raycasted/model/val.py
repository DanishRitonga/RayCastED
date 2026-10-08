"""RayCastED — Polygon Validation Metrics (Phase 8).

RayCastValidator subclasses DetectionValidator, replacing bounding-box IoU
with GPU-accelerated Polar IoU. Reports polar mAP and bPQ (from Polar IoU).

bPQ is used as the primary model selection metric (fitness).

Spec reference: docs/project.md section 15
"""

from pathlib import Path

import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class

from raycasted.data.etl.ops.iou import polar_iou_pairwise_flat_torch
from raycasted.data.etl.utils import constants as _val_const
from raycasted.data.etl.utils.constants import configure_rays
from raycasted.evals.polygon_nms import polygon_nms_indices
from raycasted.model.blocks.head import RayCastDetect
from raycasted.model.metrics import compute_bpq_from_iou

RAYCAST_DIM = 2 + _val_const.N_RAYS


# ---------------------------------------------------------------------------
# RayCastDetMetrics
# ---------------------------------------------------------------------------


class RayCastDetMetrics(DetMetrics):
    """Polygon detection metrics: polar mAP + bPQ (from Polar IoU).

    Extends DetMetrics with two metric tracks:
      - shapely: mAP computed using Polar IoU
      - bpq: binary Panoptic Quality via Polar IoU (primary fitness)

    Stats dict keys:
      - tp_shapely: Polar IoU true-positive matrix
      - conf, pred_cls, target_cls, target_img: shared across all tracks

    bPQ is accumulated per-image in bpq_sum / bpq_count (not via ap_per_class).
    """

    def __init__(self, names=None):
        super().__init__(names or {})
        self.shapely = Metric()
        self.stats = dict(tp_shapely=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.box = Metric()  # kept for DetMetrics compatibility but unused
        self.bpq_sum = 0.0
        self.bpq_count = 0
        self.bsq_sum = 0.0
        self.bdq_sum = 0.0

    def process(self, save_dir=Path('.'), plot=False, on_plot=None):
        """Compute mAP from shapely true-positive stats."""
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

        return stats

    @property
    def keys(self):
        """Return metric key names for shapely and bPQ tracks."""
        return [
            'metrics/precision(P)',
            'metrics/recall(P)',
            'metrics/mAP50(P)',
            'metrics/mAP50-95(P)',
            'metrics/bPQ',
            'metrics/bSQ',
            'metrics/bDQ',
        ]

    def mean_results(self):
        """Return mean results for shapely and bPQ tracks."""
        bpq = self.bpq_sum / max(self.bpq_count, 1)
        bsq = self.bsq_sum / max(self.bpq_count, 1)
        bdq = self.bdq_sum / max(self.bpq_count, 1)
        return self.shapely.mean_results() + [bpq, bsq, bdq]

    def class_result(self, i):
        """Return per-class results for shapely track."""
        return self.shapely.class_result(i)

    @property
    def maps(self):
        """Return mAP scores per class (shapely track)."""
        return self.shapely.maps

    @property
    def fitness(self):
        """Fitness: 0.3×mAP50 + 0.7×bPQ — weights bPQ higher for mask-quality selection."""
        mAP50 = self.shapely.mean_results()[2]
        bpq = self.bpq_sum / max(self.bpq_count, 1)
        return 0.3 * mAP50 + 0.7 * bpq

    @property
    def results_dict(self):
        """Return dict of all metrics."""
        keys = [*self.keys, 'fitness']
        values = [*self.mean_results(), self.fitness]
        return dict(zip(keys, values))

    def update_bpq(self, bpq: float, bsq: float, bdq: float):
        """Accumulate per-image bPQ components."""
        self.bpq_sum += bpq
        self.bsq_sum += bsq
        self.bdq_sum += bdq
        self.bpq_count += 1


# ---------------------------------------------------------------------------
# RayCastValidator
# ---------------------------------------------------------------------------


class RayCastValidator(DetectionValidator):
    """Polygon detection validator with GPU Polar IoU.

    Subclasses DetectionValidator, replacing bounding-box IoU with GPU-accelerated
    Polar IoU (same metric used in training loss). Reports polar mAP and bPQ.

    The standard one-to-one path is NMS-free.  When the explicit duplicate
    training experiment is enabled, class-agnostic polygon NMS metrics become
    the primary validation track and raw metrics are retained as telemetry.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        duplicate_eval=False,
        polygon_nms_iou=0.5,
    ):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = 'detect'
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.metrics = RayCastDetMetrics()
        self.duplicate_eval = bool(duplicate_eval)
        self.polygon_nms_iou = float(polygon_nms_iou)
        if not 0.0 <= self.polygon_nms_iou <= 1.0:
            raise ValueError('polygon_nms_iou must be in [0, 1]')
        self.raw_metrics = RayCastDetMetrics() if self.duplicate_eval else None
        self.raw_prediction_count = 0
        self.nms_prediction_count = 0
        self.raycast_dim = 2 + _val_const.N_RAYS  # default; overwritten in init_metrics from model head

    def get_desc(self):
        """Return a formatted header string for polygon + bPQ metrics."""
        return ('%22s' + '%11s' * 9) % (
            'Class',
            'Images',
            'Instances',
            'Poly(P',
            'R',
            'mAP50',
            'mAP50-95)',
            'bPQ',
            'bSQ',
            'bDQ',
        )

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
        self.raw_metrics = RayCastDetMetrics(names=model.names) if self.duplicate_eval else None
        self.raw_prediction_count = 0
        self.nms_prediction_count = 0
        self.confusion_matrix = None  # skip — incompatible with polygon format
        # Derive raycast_dim from model head — traverse nested model structure
        # During training: model is DetectionModel, model.model is nn.Sequential
        # During final_eval: model.model may be DetectionModel again (unwrapped)
        head = model
        while hasattr(head, 'model') and not isinstance(head, RayCastDetect):
            child = head.model
            if isinstance(child, (list, torch.nn.Sequential)):
                head = child[-1]
                break
            head = child
        self.raycast_dim = getattr(head, 'raycast_dim', 2 + _val_const.N_RAYS)
        self.n_rays = getattr(head, 'n_rays', _val_const.N_RAYS)
        self.no_object = bool(getattr(head, 'no_object', False))
        configure_rays(self.n_rays)

    def finalize_metrics(self, *args, **kwargs):
        """Skip confusion matrix plotting — incompatible with raycast polygon data."""

    def postprocess(self, preds):
        """Extract polygon predictions from model output (no NMS).

        Handles two output formats:
          - Detect head: [B, max_det, raycast_dim+2] where col raycast_dim=max_conf, raycast_dim+1=cls
          - RTDETRDecoder: [B, num_queries, raycast_dim+nc] where cols raycast_dim: are per-class scores

        Returns:
            list[dict] with keys 'bboxes' [N,raycast_dim], 'conf' [N], 'cls' [N].
        """
        pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
        n_channels = pred_tensor.shape[-1]

        if n_channels == self.raycast_dim + 2:
            # Detect head format: max_conf + cls_idx
            conf_idx = self.raycast_dim
            cls_idx = self.raycast_dim + 1
            outputs = []
            for i in range(pred_tensor.shape[0]):
                det = pred_tensor[i]
                mask = det[:, conf_idx] > self.args.conf
                outputs.append(
                    {
                        'bboxes': det[mask, : self.raycast_dim],
                        'conf': det[mask, conf_idx],
                        'cls': det[mask, cls_idx],
                    }
                )
        else:
            # RTDETRDecoder format: per-class scores. With the ∅ ("no object")
            # head the last column is P(no-object); decode matching the eval_lsp
            # protocol: drop a query iff its overall argmax is the ∅ column, and
            # use conf = 1 - P(∅) with cls = argmax over the real classes (instead
            # of a raw max over all nc+1 columns, which can select ∅ as a class).
            all_scores = pred_tensor[..., self.raycast_dim :]
            if self.no_object:
                no_obj = all_scores[..., -1]
                real_scores = all_scores[..., :-1]
                is_bg = all_scores.argmax(dim=-1) == all_scores.shape[-1] - 1
                conf = 1.0 - no_obj.sigmoid()
                cls = real_scores.argmax(dim=-1)
            else:
                conf, cls = all_scores.max(dim=-1)
                is_bg = torch.zeros_like(conf, dtype=torch.bool)
            outputs = []
            for i in range(pred_tensor.shape[0]):
                mask = (conf[i] > self.args.conf) & (~is_bg[i])
                outputs.append(
                    {
                        'bboxes': pred_tensor[i, mask, : self.raycast_dim],
                        'conf': conf[i, mask],
                        'cls': cls[i, mask],
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

    def _polygon_nms_pred(self, pred):
        """Apply class-agnostic exact polygon-IoU NMS to one image."""
        if pred['bboxes'].shape[0] == 0:
            return pred
        keep = polygon_nms_indices(
            pred['bboxes'].detach().cpu().numpy(),
            pred['conf'].detach().cpu().numpy(),
            iou_threshold=self.polygon_nms_iou,
            class_ids=pred['cls'].detach().cpu().numpy(),
            class_agnostic=True,
        )
        keep = torch.as_tensor(keep, device=pred['bboxes'].device, dtype=torch.long)
        return {key: value[keep] for key, value in pred.items()}

    def _process_batch(self, preds, batch):
        """Compute GPU Polar IoU true-positive matrix and return IoU matrix.

        Uses the same polar IoU as training loss (consistent metric) computed
        entirely on GPU. Only the final TP matrix transfers to CPU.

        Args:
            preds: dict with 'bboxes' [N_pred, 34], 'conf', 'cls'.
            batch: dict with 'bboxes' [N_gt, 34], 'cls' from _prepare_batch.

        Returns:
            dict with 'tp_shapely' array and 'iou_matrix' [N_pred, N_gt].
        """
        n_pred = preds['cls'].shape[0]
        n_gt = batch['cls'].shape[0]

        if n_gt == 0 or n_pred == 0:
            return {
                'tp_shapely': np.zeros((n_pred, self.niou), dtype=bool),
                'iou_matrix': np.zeros((n_pred, n_gt), dtype=np.float32),
            }

        # Extract rays from polygon tensors (keep on GPU)
        pred_rays = preds['bboxes'][:, 2:]  # [N_pred, n_rays]
        gt_rays = batch['bboxes'][:, 2:]  # [N_gt, n_rays]

        # Expand to pairwise shape for polar IoU on GPU
        pred_exp = pred_rays[:, None, :].expand(n_pred, n_gt, self.n_rays)
        gt_exp = gt_rays[None, :, :].expand(n_pred, n_gt, self.n_rays)
        iou_matrix = polar_iou_pairwise_flat_torch(pred_exp, gt_exp)  # [N_pred, N_gt]

        # Polar IoU true-positive matching (same metric as training loss)
        # match_predictions expects iou shape [N_gt, N_pred] (parent convention)
        tp_shapely = self.match_predictions(preds['cls'], batch['cls'], iou_matrix.T).cpu().numpy()

        return {'tp_shapely': tp_shapely, 'iou_matrix': iou_matrix.cpu().numpy()}

    def _update_metric_track(self, metrics, predn, pbatch):
        """Update one raw or NMS metric track for a single image."""
        cls = pbatch['cls'].cpu().numpy()
        no_pred = predn['cls'].shape[0] == 0
        batch_result = self._process_batch(predn, pbatch)
        metrics.update_stats(
            {
                **{k: v for k, v in batch_result.items() if k != 'iou_matrix'},
                'target_cls': cls,
                'target_img': np.unique(cls),
                'conf': np.zeros(0) if no_pred else predn['conf'].cpu().numpy(),
                'pred_cls': np.zeros(0) if no_pred else predn['cls'].cpu().numpy(),
            }
        )
        bpq, bsq, bdq = compute_bpq_from_iou(batch_result['iou_matrix'])
        metrics.update_bpq(bpq, bsq, bdq)

    def update_metrics(self, preds, batch):
        """Update metrics with polygon predictions and ground truth.

        Overridden to pass tp_shapely, compute bPQ from polar IoU matrix, and skip confusion matrix.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            if self.duplicate_eval:
                self.raw_prediction_count += int(predn['cls'].shape[0])
                self._update_metric_track(self.raw_metrics, predn, pbatch)
                predn = self._polygon_nms_pred(predn)
                self.nms_prediction_count += int(predn['cls'].shape[0])
            self._update_metric_track(self.metrics, predn, pbatch)

    def get_stats(self):
        """Compute and return validation metrics."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        results = self.metrics.results_dict
        self.metrics.clear_stats()
        if self.duplicate_eval:
            self.raw_metrics.process(save_dir=self.save_dir, plot=False, on_plot=None)
            raw_results = self.raw_metrics.results_dict
            self.raw_metrics.clear_stats()
            raw_names = {
                'metrics/precision(P)': 'metrics/raw_precision(P)',
                'metrics/recall(P)': 'metrics/raw_recall(P)',
                'metrics/mAP50(P)': 'metrics/raw_mAP50(P)',
                'metrics/mAP50-95(P)': 'metrics/raw_mAP50-95(P)',
                'metrics/bPQ': 'metrics/raw_bPQ',
                'metrics/bSQ': 'metrics/raw_bSQ',
                'metrics/bDQ': 'metrics/raw_bDQ',
            }
            results.update({raw_names[key]: value for key, value in raw_results.items() if key in raw_names})
            results['metrics/raw_predictions'] = self.raw_prediction_count
            results['metrics/nms_predictions'] = self.nms_prediction_count
            results['metrics/nms_keep_ratio'] = self.nms_prediction_count / max(self.raw_prediction_count, 1)
        return results

    def plot_predictions(self, batch, preds, ni):
        """Skip standard bbox prediction plotting — incompatible with 34-dim polygon data."""

    def plot_val_samples(self, batch, ni):
        """Skip standard bbox val sample plotting — incompatible with 34-dim polygon data."""
