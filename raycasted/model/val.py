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
from raycasted.model.blocks.head import RayCastDetect
from raycasted.model.blocks.rtdetr_head import RayCastRTDETRDecoder
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
        """Fitness based on mAP50-95 (stable for early stopping)."""
        return self.shapely.fitness()

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

    No NMS — the end-to-end model already does top-k selection via
    Hungarian matching. Distance-based dedup is handled by RayCastPredictor.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = 'detect'
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.metrics = RayCastDetMetrics()
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
        self.confusion_matrix = None  # skip — incompatible with polygon format
        # Derive raycast_dim from model head — traverse nested model structure
        # During training: model is DetectionModel, model.model is nn.Sequential
        # During final_eval: model.model may be DetectionModel again (unwrapped)
        head = model
        while hasattr(head, 'model') and not isinstance(head, (RayCastDetect, RayCastRTDETRDecoder)):
            child = head.model
            if isinstance(child, (list, torch.nn.Sequential)):
                head = child[-1]
                break
            head = child
        self.raycast_dim = getattr(head, 'raycast_dim', 2 + _val_const.N_RAYS)
        self.n_rays = getattr(head, 'n_rays', _val_const.N_RAYS)
        configure_rays(self.n_rays)

    def finalize_metrics(self, *args, **kwargs):
        """Skip confusion matrix plotting — incompatible with raycast polygon data."""

    def postprocess(self, preds):
        """Extract polygon predictions from model output (no NMS).

        Handles three output formats:
          - Detect head: [B, max_det, raycast_dim+2] where col raycast_dim=max_conf, raycast_dim+1=cls
          - RTDETRDecoder: [B, num_queries, raycast_dim+nc] where cols raycast_dim: are per-class scores
          - HybridDecoder dict: {'pred_logits':[B,Q,nc+1], 'pred_points':[B,Q,2], 'pred_radial':[B,Q,n_rays]}

        Returns:
            list[dict] with keys 'bboxes' [N,raycast_dim], 'conf' [N], 'cls' [N].
        """
        if isinstance(preds, dict):
            return self._postprocess_hybrid(preds)

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
            # RTDETRDecoder format: per-class scores
            cls_scores = pred_tensor[..., self.raycast_dim :]
            conf, cls = cls_scores.max(dim=-1)
            outputs = []
            for i in range(pred_tensor.shape[0]):
                mask = conf[i] > self.args.conf
                outputs.append(
                    {
                        'bboxes': pred_tensor[i, mask, : self.raycast_dim],
                        'conf': conf[i, mask],
                        'cls': cls[i, mask],
                    }
                )
        return outputs

    def _postprocess_hybrid(self, preds: dict[str, torch.Tensor]) -> list[dict]:
        """Convert hybrid decoder output dict to validator format.

        Hybrid decoder returns:
          - pred_logits: [B, Q, nc+1]  no-object at last position
          - pred_points: [B, Q, 2]     centroids in [0,1] normalised
          - pred_radial: [B, Q, n_rays]  log-space ray distances

        Converts to pixel-space polygon vectors and extracts class confidences.

        Returns:
            list[dict] with keys 'bboxes' [N,raycast_dim], 'conf' [N], 'cls' [N].
        """
        logits = preds['pred_logits']  # [B, Q, nc+1]
        points = preds['pred_points']  # [B, Q, 2]  normalised [0,1]
        radial = preds['pred_radial']  # [B, Q, n_rays]  log-space

        # Use no-object class to suppress bg queries at inference.
        # The model outputs nc+1 logits where position nc is the no-object class.
        # Queries where no-object has the highest logit are suppressed.
        # Matches LSP-DETR inference: argmax(logits) != num_classes.
        no_object_idx = logits.shape[-1] - 1  # = nc
        cls_logits = logits[..., :-1]  # [B, Q, nc]
        cls_prob = cls_logits.softmax(dim=-1)
        conf, cls = cls_prob.max(dim=-1)

        _inference_conf = getattr(self.args, 'conf', 0.25)
        is_object = logits.argmax(dim=-1) != no_object_idx  # [B, Q]
        keep = (conf > _inference_conf) & is_object

        # Convert to pixel space (crop_size = 256)
        imgsz = self.args.imgsz
        points_px = points * imgsz  # [B, Q, 2]
        rays_px = radial.exp() * imgsz  # [B, Q, n_rays]

        bboxes = torch.cat([points_px, rays_px], dim=-1)  # [B, Q, 2+n_rays]

        outputs = []
        for i in range(bboxes.shape[0]):
            mask = keep[i]
            outputs.append(
                {
                    'bboxes': bboxes[i, mask],
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

    def update_metrics(self, preds, batch):
        """Update metrics with polygon predictions and ground truth.

        Overridden to pass tp_shapely, compute bPQ from polar IoU matrix, and skip confusion matrix.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch['cls'].cpu().numpy()
            no_pred = predn['cls'].shape[0] == 0
            batch_result = self._process_batch(predn, pbatch)
            self.metrics.update_stats(
                {
                    **{k: v for k, v in batch_result.items() if k != 'iou_matrix'},
                    'target_cls': cls,
                    'target_img': np.unique(cls),
                    'conf': np.zeros(0) if no_pred else predn['conf'].cpu().numpy(),
                    'pred_cls': np.zeros(0) if no_pred else predn['cls'].cpu().numpy(),
                }
            )

            # bPQ via polar IoU matrix (same metric as training, no rasterization)
            iou_matrix = batch_result['iou_matrix']
            bpq, bsq, bdq = compute_bpq_from_iou(iou_matrix)
            self.metrics.update_bpq(bpq, bsq, bdq)

    def get_stats(self):
        """Compute and return validation metrics."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def plot_predictions(self, batch, preds, ni):
        """Skip standard bbox prediction plotting — incompatible with 34-dim polygon data."""

    def plot_val_samples(self, batch, ni):
        """Skip standard bbox val sample plotting — incompatible with 34-dim polygon data."""
