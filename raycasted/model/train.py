"""RayCastED — Training Pipeline (Phase 10).

Wires all model components into the Ultralytics training loop:
  - RayCastDetect head (replaces standard Detect)
  - RayCastE2ELoss (4-term polygon loss with smoothness annealing)
  - RayCastTileDataset (custom .npz data loading)
  - RayCastValidator (Shapely polygon mAP)

Usage:
    from raycasted.model.train import RayCastTrainer

    trainer = RayCastTrainer(overrides={
        'model': 'yolo26s.yaml',
        'data': 'configs/dataset.yaml',
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
    })
    trainer.train()
"""

import copy
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import initialize_weights

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.utils import constants as _const
from raycasted.model.blocks.head import RayCastDetect
from raycasted.model.blocks.hybrid_decoder import HybridRayCastDecoder
from raycasted.model.blocks.rtdetr_head import RayCastRTDETRDecoder
from raycasted.model.builder import raycasted_parse_model
from raycasted.model.loss import RayCastE2ELoss
from raycasted.model.register import register_raycast_head
from raycasted.model.val import RayCastValidator


def _extract_neck_channels(head) -> tuple:
    """Extract neck output channel sizes from a Detect head.

    Inspects the first conv in each scale branch of the box regression
    stack to recover the neck feature map channels.

    Args:
        head: A Detect (or subclass) head module.

    Returns:
        Tuple of channel sizes, one per feature scale.
    """
    return tuple(head.cv2[i][0].conv.in_channels for i in range(head.nl))


def _load_backbone_weights(model, weights_path: str) -> None:
    """Load pretrained backbone weights, skipping neck and head.

    Loads only layers 0-10 (backbone: P1/2 through P5/32 + SPPF + C2PSA).
    Neck and head are left randomly initialised so they can have any
    architecture (P2-P4, DWT, etc.) without weight mismatch.

    Args:
        model: DetectionModel (returned by super().get_model()).
        weights_path: Path to pretrained .pt file (e.g. 'yolo26s.pt').
    """
    from ultralytics import YOLO

    pretrained = YOLO(weights_path)
    pretrained_state = pretrained.model.state_dict()
    backbone_prefixes = tuple(f'model.{i}.' for i in range(11))
    backbone_state = {k: v for k, v in pretrained_state.items() if k.startswith(backbone_prefixes)}

    model_state = model.state_dict()
    matched = {k: v for k, v in backbone_state.items() if k in model_state and model_state[k].shape == v.shape}
    skipped = {k: v for k, v in backbone_state.items() if k not in matched}

    model_state.update(matched)
    model.load_state_dict(model_state)

    n_matched = sum(v.numel() for v in matched.values())
    n_skipped = sum(v.numel() for v in skipped.values())
    print(
        f'Pretrained backbone: {len(matched)}/{len(backbone_state)} keys loaded '
        f'({n_matched:,} params), {len(skipped)} skipped ({n_skipped:,} params)'
    )


def _reconstruct_polygon_vertices(bboxes_norm: torch.Tensor, crop_size: float):
    """Reconstruct pixel-space polygon vertices from normalized ray vectors.

    Args:
        bboxes_norm: [N, 2+n_rays] — (cx, cy, r0..r63) in [0, 1]
        crop_size: image size in pixels

    Returns:
        vertices: [N, n_rays, 2] pixel-space vertex coordinates
    """
    from raycasted.data.etl.utils.constants import N_RAYS

    cx = bboxes_norm[:, 0] * crop_size
    cy = bboxes_norm[:, 1] * crop_size
    rays_norm = bboxes_norm[:, 2 : 2 + N_RAYS]
    rays_px = rays_norm * crop_size

    from raycasted.data.etl.utils import constants as _const

    cos = torch.from_numpy(_const.RAY_COS).float()
    sin = torch.from_numpy(_const.RAY_SIN).float()
    vx = cx.unsqueeze(1) + rays_px * cos.unsqueeze(0)
    vy = cy.unsqueeze(1) + rays_px * sin.unsqueeze(0)
    return torch.stack([vx, vy], dim=-1)


def _raycast_collate_fn(batch: list) -> dict:
    """Collate (image, labels) tuples into a batch dict for training.

    Produces the dict format expected by RayCastDetectionLoss:
        'img':        [B, 3, H, W] float32
        'batch_idx':  [sum_M] float32
        'cls':        [sum_M] float32 (class id)
        'bboxes':     [sum_M, raycast_dim] float32 (cx, cy, d_1..d_n)

    Args:
        batch: List of (image_tensor, labels_array) tuples from RayCastTileDataset.

    Returns:
        Dict with img, batch_idx, cls, bboxes tensors.
    """
    images = torch.stack([item[0] for item in batch])

    # Unpack items — handle both 2-tuple and 3-tuple formats
    labels_list = [item[1] for item in batch]

    target_list = []
    for batch_idx, labels in enumerate(labels_list):
        if labels.shape[0] == 0:
            continue
        batch_col = np.full((labels.shape[0], 1), batch_idx, dtype=np.float32)
        target_list.append(np.concatenate([batch_col, labels], axis=1))

    if target_list:
        targets = torch.from_numpy(np.concatenate(target_list, axis=0))
    else:
        # Empty batch — shape must match 2 + n_rays (cx, cy, d_1..d_n) for bboxes
        from raycasted.data.etl.utils.constants import N_RAYS as _n_rays  # noqa: N811

        targets = torch.zeros((0, 4 + _n_rays), dtype=torch.float32)

    # Metadata required by validator (ori_shape, ratio_pad, im_file)
    ori_shapes = []
    ratio_pads = []
    im_files = []
    for item_idx in range(len(batch)):
        img = images[item_idx]
        h, w = img.shape[1:]  # CHW format
        ori_shapes.append(torch.tensor([h, w]))
        ratio_pads.append((torch.ones(1, 1), torch.zeros(1, 2)))  # identity transform
        im_files.append(f'tile_{item_idx:04d}.npz')

    bboxes = targets[:, 2:]  # [N, 2+n_rays] = cx, cy, d_1..d_n (normalized)
    crop_size = float(images.shape[2])

    gt_vertices = _reconstruct_polygon_vertices(bboxes, crop_size)

    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': bboxes,
        'gt_vertices': gt_vertices,  # [N, n_rays, 2] pixel-space vertices
        'ori_shape': torch.stack(ori_shapes),
        'ratio_pad': ratio_pads,
        'im_file': im_files,
    }


class _RayCastCriterionWrapper:
    """Picklable callable that wraps RayCastE2ELoss creation.

    Replaces a lambda so that torch.save / torch.load can serialize the model.
    Called as: model.init_criterion() — the wrapper is callable.
    """

    def __init__(self, model, max_epochs: int = 200, training_config: dict | None = None, steps_per_epoch: int = 0):
        self._model = model
        self._max_epochs = max_epochs
        self._training_config = training_config
        self._steps_per_epoch = steps_per_epoch

    def _build_class_weights(self, tcfg: dict):
        """Build per-class inverse-frequency weights from config or training data.

        Uses ``class_weights`` from config. Supports:
          - None: no weighting (default)
          - 'auto': compute from training data class distribution (requires
            trainer to call set_class_weights later)
          - list[float]: explicit per-class weights

        Returns:
            torch.Tensor [nc] or None.
        """
        import torch as _torch

        cw = tcfg.get('class_weights')
        if cw is None:
            return None
        if isinstance(cw, str) and cw == 'auto':
            # Placeholder — will be replaced by set_class_weights after data setup
            return None
        if isinstance(cw, (list, tuple)):
            return _torch.tensor(cw, dtype=_torch.float32)
        return None

    def __call__(self):
        tcfg = self._training_config or {}
        return RayCastE2ELoss(
            self._model,
            max_epochs=self._max_epochs,
            tal_topk=tcfg.get('tal_topk', 13),
            assigner_radius_scale=tcfg.get('assigner_radius_scale', 1.5),
            assigner_alpha=tcfg.get('assigner_alpha', 0.5),
            assigner_beta=tcfg.get('assigner_beta', 6.0),
            nwd_enabled=tcfg.get('nwd_enabled', False),
            nwd_c=tcfg.get('nwd_c', 0.001),
            log_ray_loss=tcfg.get('log_ray_loss', False),
            focal_gamma=tcfg.get('focal_gamma', 0.0),
            focal_alpha=tcfg.get('focal_alpha', 0.25),
            align_threshold=tcfg.get('align_threshold', 0.0),
            gradnorm=tcfg.get('gradnorm', False),
            gradnorm_alpha=tcfg.get('gradnorm_alpha', 0.5),
            gradnorm_warmup_epochs=tcfg.get('gradnorm_warmup_epochs', 5),
            lambda_aux_xy=tcfg.get('aux_xy_weight', 0.0),
            aux_xy_ramp_epochs=tcfg.get('aux_xy_ramp_epochs', 100),
            bg_fg_ratio=tcfg.get('bg_fg_ratio', 3),
            ohem_bg_ratio=tcfg.get('ohem_bg_ratio', 0.0),
            plb_enabled=tcfg.get('plb_enabled', False),
            plb_cls_weight=tcfg.get('plb_cls_weight', 1.0),
            bg_cls_decay=tcfg.get('bg_cls_decay', 1.0),
            fg_cls_boost=tcfg.get('fg_cls_boost', 0.0),
            soft_targets=tcfg.get('soft_targets', False),
            soft_targets_o2o=tcfg.get('soft_targets_o2o', None),
            focal_gamma_o2o=tcfg.get('focal_gamma_o2o', None),
            focal_alpha_o2o=tcfg.get('focal_alpha_o2o', None),
            bg_fg_ratio_o2o=tcfg.get('bg_fg_ratio_o2o', None),
            ohem_bg_ratio_o2o=tcfg.get('ohem_bg_ratio_o2o', None),
            bg_cls_decay_o2o=tcfg.get('bg_cls_decay_o2o', None),
            fg_cls_boost_o2o=tcfg.get('fg_cls_boost_o2o', None),
            class_weights=self._build_class_weights(tcfg),
            o2o_topk2_start=tcfg.get('o2o_topk2_start', 1),
            o2o_topk2_anneal_epoch=tcfg.get('o2o_topk2_anneal_epoch', 0),
            o2o_topk2_anneal_end=tcfg.get('o2o_topk2_anneal_end', 0.5),
            sigma_anneal_start=tcfg.get('sigma_anneal_start', 0.0),
            sigma_anneal_end=tcfg.get('sigma_anneal_end', 0.0),
            sigma_anneal_epoch=tcfg.get('sigma_anneal_epoch', 0),
            stal_min_positives=tcfg.get('stal_min_positives', 0),
            lambda_l1=tcfg.get('lambda_l1', 14.0),
            lambda_piou=tcfg.get('lambda_piou', 13.0),
            lambda_cls=tcfg.get('lambda_cls', 2.0),
            lambda_xy=tcfg.get('lambda_xy', 500.0),
            fg_cls_quality_scale=tcfg.get('fg_cls_quality_scale', 0.0),
            fg_cls_quality_scale_o2o=tcfg.get('fg_cls_quality_scale_o2o', None),
            steps_per_epoch=self._steps_per_epoch,
            gaussian_soft_targets=tcfg.get('gaussian_soft_targets', False),
            gaussian_sigma=tcfg.get('gaussian_sigma', 0.5),
            prediction_refinement_weight=tcfg.get('prediction_refinement_weight', 0.0),
            range_l1_weight=tcfg.get('range_l1_weight', 0.0),
            range_l1_eps=tcfg.get('range_l1_eps', 0.1),
            bound_l1_weight=tcfg.get('bound_l1_weight', 0.0),
            bound_l1_eps=tcfg.get('bound_l1_eps', 0.1),
            hierarchical_cls=tcfg.get('hierarchical_cls', False),
            nc_override=tcfg.get('nc_override', None),
            cls_only_tal=tcfg.get('cls_only_tal', False),
            o2o_distill_weight=tcfg.get('o2o_distill_weight', 0.0),
        )


class RayCastDetectionModel(DetectionModel):
    """Custom DetectionModel using raycasted_parse_model builder.

    Replaces ultralytics' parse_model() with our custom builder that
    supports custom blocks (ResoConv, C3k2_LK) without monkey-patching.

    All other DetectionModel behavior (stride computation, bias_init,
    init_criterion, etc.) is preserved via inheritance from BaseModel.
    """

    def __init__(self, cfg='yolo26s.yaml', ch=3, nc=None, verbose=True):
        """Initialize the YOLO detection model with custom builder.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        from ultralytics.nn.tasks import yaml_model_load
        from ultralytics.utils import LOGGER

        super(DetectionModel, self).__init__()  # BaseModel.__init__ only — skip parse_model

        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if self.yaml['backbone'][0][2] == 'Silence':
            LOGGER.warning(
                'YOLOv9 Silence module is deprecated in favor of torch.nn.Identity. '
                'Please delete local *.pt file and re-download the latest model checkpoint.'
            )
            self.yaml['backbone'][0][2] = 'nn.Identity'

        self.yaml['channels'] = ch
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f'Overriding model.yaml nc={self.yaml["nc"]} with nc={nc}')
            self.yaml['nc'] = nc

        self.model, self.save = raycasted_parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)

        # Build strides (same logic as DetectionModel.__init__)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x):
                output = self.forward(x)
                if self.end2end:
                    output = output['one2many']
                return output['feats']

            self.model.eval()
            m.training = True
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')


def _gradnorm_update_callback(trainer):
    """Update GradNorm weights after each training step.

    Called via on_train_batch_end.  Accesses the GradNormManager through
    the criterion (RayCastE2ELoss) attached to the model.
    """
    criterion = getattr(trainer, 'criterion', None)
    if criterion is None:
        return
    gn = getattr(criterion, 'gradnorm_manager', None)
    if gn is not None:
        gn.update()


def _best_epoch_callback(trainer):
    """Print best-epoch marker after each validation epoch.

    Compares current fitness to best_fitness and logs whether this epoch
    is the new best, or reminds what the best epoch was.
    """
    import logging

    logger = logging.getLogger('raycasted.train')
    epoch = trainer.epoch + 1  # 1-based
    fitness = trainer.fitness
    best = trainer.best_fitness

    if best is None or fitness is None:
        return

    if fitness >= best:
        logger.info(f'  ⭐ Epoch {epoch} — new best (fitness={fitness:.4f})')
    else:
        # Find best epoch from CSV (last column with max fitness)
        try:
            import csv

            best_ep = epoch  # fallback
            best_fit = best
            csv_path = trainer.csv
            if csv_path.exists():
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        val = float(row.get('fitness', 0))
                        if val >= best_fit:
                            best_fit = val
                            best_ep = int(float(row.get('epoch', epoch)))
            logger.info(f'  Best so far: epoch {best_ep} (fitness={best_fit:.4f})')
        except Exception:
            logger.info(f'  Best so far: fitness={best:.4f}')


def _lr_log_callback(trainer):
    """Log per-param-group learning rates at each epoch boundary.

    Critical for MuSGD which maintains separate LR schedules per param group
    (backbone, head, cls_head, aux_xy). Without this, LR divergence between
    groups is invisible and hard to debug.
    """
    from ultralytics.utils import LOGGER

    pg_lrs = [pg['lr'] for pg in trainer.optimizer.param_groups]
    pg_names = [pg.get('name', f'pg{i}') for i, pg in enumerate(trainer.optimizer.param_groups)]
    lr_str = ', '.join(f'{n}={lr:.2e}' for n, lr in zip(pg_names, pg_lrs))
    LOGGER.info(f'Epoch {trainer.epoch + 1} LR: {lr_str}')


def _lsp_lr_callback(trainer):
    """Replace Ultralytics' one_cycle scheduler with LSP-DETR exact schedule.

    LSP-DETR uses timm CosineLRScheduler: warmup 1e-7→1e-4 over 10 epochs,
    then pure cosine decay 1e-4→1e-6 over remaining epochs.  Ultralytics'
    one_cycle schedule has a different trajectory (~10 % lower mid-training).

    Uses a single LambdaLR with epoch-based formula — avoids SequentialLR
    milestone/off-by-one issues with Ultralytics' per-epoch stepping.

    Called via ``on_pretrain_routine_end`` (after optimizer exists).
    """
    tcfg = trainer.training_config or {}
    lr0 = tcfg.get('lr0', 1e-4)
    lr_min = tcfg.get('lrf', 0.01) * lr0  # default 1e-6
    warmup_epochs = tcfg.get('warmup_epochs', 10)
    max_epochs = getattr(trainer.args, 'epochs', 130)
    decay_epochs = max_epochs - warmup_epochs

    import math as _math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(1e-7 / lr0 + epoch * (1.0 - 1e-7 / lr0) / warmup_epochs)
        t = (epoch - warmup_epochs) / decay_epochs
        return float((_math.cos(_math.pi * t) + 1) / 2 * (1.0 - lr_min / lr0) + lr_min / lr0)

    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda)


def _hybrid_freeze_callback(trainer):
    """Freeze backbone for first N epochs (LSP-DETR-style decoder warmup).

    Handles both YOLO-style sequential models (model.model[:9]) and
    LSP-DETR-style models (model.backbone).  When unfreezing, applies
    a reduced LR (``backbone_lr_ratio``) matching PyTorch Lightning's
    BackboneFinetuning.

    Called via ``on_train_epoch_start``.
    """
    n_freeze = getattr(trainer, '_hybrid_freeze_epochs', 0)
    backbone_lr_ratio = getattr(trainer, '_hybrid_backbone_lr_ratio', 0.1)
    epoch = trainer.epoch

    model = trainer.model
    if hasattr(model, 'backbone') and not hasattr(model, 'model'):
        backbone_params = list(model.backbone.parameters())
    elif hasattr(model, 'model') and hasattr(model.model, '__getitem__'):
        backbone_params = list(model.model[:9].parameters())
    else:
        return

    if epoch < n_freeze:
        for param in backbone_params:
            param.requires_grad_(False)
    elif epoch == n_freeze:
        for param in backbone_params:
            param.requires_grad_(True)

        if trainer.optimizer is not None:
            backbone_ids = {id(p) for p in backbone_params}
            for pg in trainer.optimizer.param_groups:
                if any(id(p) in backbone_ids for p in pg['params']):
                    pg['lr'] = pg['lr'] * backbone_lr_ratio

            tcfg = trainer.training_config or {}
            lr0 = tcfg.get('lr0', 1e-4)
            lr_min = tcfg.get('lrf', 0.01) * lr0
            max_epochs = getattr(trainer.args, 'epochs', 130)
            remaining_epochs = max_epochs - epoch

            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=remaining_epochs,
                eta_min=lr_min,
                last_epoch=-1,
            )


def _is_rtdetr_yaml(cfg) -> bool:
    """Check if a model YAML specifies RayCastRTDETRDecoder as the head."""
    from ultralytics.nn.tasks import yaml_model_load

    if cfg is None:
        return False
    yaml_dict = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
    return any(len(layer) >= 3 and layer[2] == 'RayCastRTDETRDecoder' for layer in yaml_dict.get('head', []))


def _is_hybrid_yaml(cfg) -> bool:
    """Check if a model YAML specifies HybridRayCastDecoder as the head."""
    from ultralytics.nn.tasks import yaml_model_load

    if cfg is None:
        return False
    yaml_dict = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
    return any(len(layer) >= 3 and layer[2] == 'HybridRayCastDecoder' for layer in yaml_dict.get('head', []))


def _is_lsp_yaml(cfg) -> bool:
    """Check if a model YAML specifies LSP-DETR (LSPDetrModel as the head)."""
    from ultralytics.nn.tasks import yaml_model_load

    if cfg is None:
        return False
    yaml_dict = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
    return any(len(layer) >= 3 and layer[2] == 'LSPDetrModel' for layer in yaml_dict.get('head', []))


class RayCastTrainer(DetectionTrainer):
    """Training pipeline for RayCastED polygon detection model.

    Subclasses DetectionTrainer to replace the standard Detect head with
    RayCastDetect, wire RayCastE2ELoss, use RayCastTileDataset for data
    loading, and RayCastValidator for evaluation.

    Key overrides:
        get_model()          — Head replacement + loss wiring
        get_validator()      — RayCastValidator with polygon metrics
        build_dataset()      — RayCastTileDataset from .npz tiles
        get_dataloader()     — Custom collate function
        set_model_attributes — Training metadata for inference/export
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, training_config=None):
        """Initialise RayCastTrainer with mosaic/mixup forced off.

        Args:
            cfg: Default configuration.
            overrides: Dict of parameter overrides.
            _callbacks: Callback functions.
            training_config: Dict of training settings from YAML (or None for legacy defaults).
        """
        overrides = overrides or {}
        super().__init__(cfg, overrides, _callbacks)
        # Force mosaic/mixup off — they corrupt polygon targets (GAP-06)
        self.args.mosaic = 0.0
        self.args.mixup = 0.0
        self.training_config = training_config  # dict or None

        # --- Hybrid model: override optimizer for transformer decoder ---
        tcfg = self.training_config or {}
        model_yaml = overrides.get('model', '')
        if model_yaml:
            from ultralytics.nn.tasks import yaml_model_load

            yaml_dict = yaml_model_load(model_yaml)
            if _is_hybrid_yaml(yaml_dict) or _is_lsp_yaml(yaml_dict) or _is_rtdetr_yaml(yaml_dict):
                self.args.optimizer = tcfg.get('optimizer', 'AdamW')
                self.args.lr0 = tcfg.get('lr0', 1e-4)
                self.args.weight_decay = tcfg.get('weight_decay', 1e-4)
                self.args.warmup_epochs = tcfg.get('warmup_epochs', 10)
                self.args.warmup_bias_lr = 0.0  # no special bias lr: all params warm from zero
                self.args.pretrained = tcfg.get('pretrained', False)
                # Register backbone freeze callback for hybrid/lsp training
                freeze_epochs = tcfg.get('backbone_freeze_epochs', 0)
                if freeze_epochs > 0:
                    self._hybrid_freeze_epochs = freeze_epochs
                    self._hybrid_backbone_lr_ratio = tcfg.get('backbone_lr_ratio', 0.1)
                    self.add_callback('on_train_epoch_start', _hybrid_freeze_callback)

            # LSP-DETR: replace Ultralytics one_cycle with exact LSP-DETR cosine schedule
            if _is_lsp_yaml(yaml_dict):
                self.args.cos_lr = False  # disable one_cycle; _lsp_lr_callback handles it
                self.add_callback('on_pretrain_routine_end', _lsp_lr_callback)

        # Register GradNorm callback — updates dynamic loss weights after each step
        self.add_callback('on_train_batch_end', _gradnorm_update_callback)
        # Register best-epoch logger — prints after each validation epoch
        self.add_callback('on_fit_epoch_end', _best_epoch_callback)
        # Register per-epoch LR logging — critical for MuSGD multi-group debugging
        self.add_callback('on_fit_epoch_end', _lr_log_callback)

    def plot_training_labels(self):
        """Skip standard bbox label plotting — incompatible with raycast polygon data."""

    def optimizer_step(self):
        """Override gradient clipping to use configurable max_norm (LSP-DETR uses 0.1).

        Base class hardcodes max_norm=10.0 which is too loose for piou loss
        gradients that can explode to ~10^7 in early training (AGENTS.md #23).
        """
        self.scaler.unscale_(self.optimizer)
        tcfg = self.training_config or {}
        max_norm = tcfg.get('clip_grad', 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def _setup_scheduler(self):
        """Override scheduler setup for CosineAnnealingWarmRestarts (SGDR).

        When warm_restarts=True, uses PyTorch's CosineAnnealingWarmRestarts
        which resets LR to lr0 at each period boundary (T_0, T_0+T_0*T_mult, ...).
        This prevents the recall collapse seen with single-cosine decay where
        LR drops to <17% of peak by epoch 220 (60% of training wasted).

        Schedule with T_0=200, T_mult=2, 600 epochs:
          Cycle 1: epochs 0-199 (LR: lr0 → eta_min)
          Cycle 2: epochs 200-599 (LR: lr0 → eta_min, longer second cycle)
          Total: 2 restarts, each allowing model to recover exploration.

        When warm_restarts=False, falls back to base class (single cosine).
        """
        tcfg = self.training_config or {}
        if tcfg.get('warm_restarts', False):
            t0 = tcfg.get('warm_restarts_T0', 200)
            t_mult = tcfg.get('warm_restarts_T_mult', 2)
            eta_min = tcfg.get('warm_restarts_eta_min', 0.0001)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t0,
                T_mult=t_mult,
                eta_min=eta_min,
            )
            self.lf = lambda epoch: max(
                (1 + math.cos(math.pi * epoch / t0)) / 2,
                eta_min / self.args.lr0,
            )
        else:
            super()._setup_scheduler()

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create YOLO model with RayCastDetect head and RayCastE2ELoss.

        Uses RayCastDetectionModel (custom builder) instead of standard
        DetectionModel. Replaces the Detect head with RayCastDetect and
        patches init_criterion to return RayCastE2ELoss.

        If the YAML contains RayCastRTDETRDecoder, creates
        RayCastRTDETRDetectionModel instead (transformer decoder with
        ray polygon regression).

        Args:
            cfg: Model config path or YAML name.
            weights: Pretrained weights path or checkpoint model object (on resume).
            verbose: Print model info.

        Returns:
            RayCastDetectionModel or RayCastRTDETRDetectionModel.
        """
        register_raycast_head()
        nc = self.data.get('nc') if hasattr(self, 'data') and self.data else None
        tcfg = self.training_config or {}
        nc_override = tcfg.get('nc_override') if tcfg else None
        if nc_override is not None and nc is not None and nc_override != nc:
            from ultralytics.utils import LOGGER

            LOGGER.info(f'nc_override={nc_override}: overriding data nc={nc} → {nc_override}')
            nc = nc_override
            self.data['nc'] = nc_override
            self.data['names'] = {i: f'class_{i}' for i in range(nc_override)}

        # Detect RT-DETR YAML — if head contains RayCastRTDETRDecoder,
        # use the RT-DETR model class (different loss, no E2E head patching)
        is_rtdetr = _is_rtdetr_yaml(cfg)
        is_hybrid = _is_hybrid_yaml(cfg)
        is_lsp = _is_lsp_yaml(cfg)

        _const.configure_rays(self.training_config.get('n_rays', 64) if self.training_config else 64)

        if is_lsp:
            from raycasted.model.lsp_detr_model import LSPDetrDetectionModel

            if weights is not None and isinstance(weights, LSPDetrDetectionModel):
                weights.end2end = False
                lsp_yaml = {'nc': nc, 'head': [[[2, 4, 8], 1, 'LSPDetrModel', ['nc', 64]]]}
                weights.yaml = weights.yaml if hasattr(weights, 'yaml') else lsp_yaml
                return weights  # resume: keep the loaded checkpoint with trained weights
            model = LSPDetrDetectionModel(nc=nc, n_rays=_const.N_RAYS)
            return model
        elif is_hybrid:
            from raycasted.model.hybrid_model import HybridDetectionModel

            model = HybridDetectionModel(cfg, ch=3, nc=nc, verbose=verbose)
            return model
        elif is_rtdetr:
            from raycasted.model.rtdetr_model import RayCastRTDETRDetectionModel

            model = RayCastRTDETRDetectionModel(cfg, ch=3, nc=nc, verbose=verbose)
            return model
        else:
            model = RayCastDetectionModel(cfg, ch=3, nc=nc, verbose=verbose)

        # On resume, `weights` is the checkpoint model object (from load_checkpoint).
        # RayCastDetectionModel.__init__ creates a fresh model from YAML, losing
        # any extra modules (attention, quality head) that were attached after
        # construction. Save the checkpoint head's state dict to restore later.
        checkpoint_head_sd = None
        if weights is not None and isinstance(weights, torch.nn.Module):
            ckpt_head = None
            for m in weights.model.children():
                if isinstance(m, RayCastDetect):
                    ckpt_head = m
                    break
            if ckpt_head is not None:
                checkpoint_head_sd = ckpt_head.state_dict()

        # Replace Detect head → RayCastDetect
        old_head = model.model[-1]
        tcfg = self.training_config
        aux_xy = bool(tcfg.get('aux_xy_weight', 0) > 0) if tcfg else False
        prediction_refinement = bool(tcfg.get('prediction_refinement_weight', 0) > 0) if tcfg else False
        prediction_refinement_topk = tcfg.get('prediction_refinement_topk', 100) if tcfg else 100
        inter_scale_competition = bool(tcfg.get('inter_scale_competition', False)) if tcfg else False
        inter_scale_temperature = tcfg.get('inter_scale_temperature', 1.0) if tcfg else 1.0
        local_competition = bool(tcfg.get('local_competition', False)) if tcfg else False
        local_competition_kernel = tcfg.get('local_competition_kernel', 3) if tcfg else 3
        local_competition_temperature = tcfg.get('local_competition_temperature', 1.0) if tcfg else 1.0
        cls_channel_scale = tcfg.get('cls_channel_scale', 1.0) if tcfg else 1.0
        cls_channel_min = tcfg.get('cls_channel_min', 0) if tcfg else 0
        dcn_in_reg_head = bool(tcfg.get('dcn_in_reg_head', False)) if tcfg else False
        dcn_in_cls_head = bool(tcfg.get('dcn_in_cls_head', False)) if tcfg else False
        hierarchical_cls = bool(tcfg.get('hierarchical_cls', False)) if tcfg else False
        hierarchical_cls_detach = bool(tcfg.get('hierarchical_cls_detach', True)) if tcfg else True
        hierarchical_binary_threshold = float(tcfg.get('hierarchical_binary_threshold', 0.01)) if tcfg else 0.01
        local_window_attn = bool(tcfg.get('local_window_attn', False)) if tcfg else False

        if isinstance(old_head, RayCastDetect):
            # YAML already specifies RayCastDetect — ensure n_rays matches
            # the configured value (YAML args don't include n_rays, so it
            # may have been created with the wrong default).
            if old_head.n_rays != _const.N_RAYS:
                old_head.n_rays = _const.N_RAYS
                old_head.raycast_dim = 2 + _const.N_RAYS
                old_head.no = old_head.nc + old_head.raycast_dim

            # Rebuild cv3 with scaled cls channels if configured differently
            c3_original = max(old_head.cv3[0][-1].in_channels, old_head.nc)
            _scale_c3 = cls_channel_scale != 1.0 or cls_channel_min > 0
            c3_new = max(cls_channel_min, int(c3_original * cls_channel_scale)) if _scale_c3 else c3_original
            if c3_new != c3_original:
                from ultralytics.nn.modules.conv import Conv as _Conv
                from ultralytics.nn.modules.conv import DWConv

                neck_ch = tuple(old_head.cv2[i][0].conv.in_channels for i in range(old_head.nl))
                old_head.cv3 = nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(DWConv(x, x, 3), _Conv(x, c3_new, 1)),
                        nn.Sequential(DWConv(c3_new, c3_new, 3), _Conv(c3_new, c3_new, 1)),
                        nn.Conv2d(c3_new, old_head.nc, 1),
                    )
                    for x in neck_ch
                )
                if old_head._end2end_arg:
                    old_head.one2one_cv3 = copy.deepcopy(old_head.cv3)

            # Set inter-scale competition flags on existing RayCastDetect head
            if inter_scale_competition:
                old_head.inter_scale_competition = True
                old_head.inter_scale_temperature = inter_scale_temperature

            # Set local competition flags on existing RayCastDetect head
            if local_competition:
                old_head.local_competition = True
                old_head.local_competition_kernel = local_competition_kernel
                old_head.local_competition_temperature = local_competition_temperature

            # Attach auxiliary xy head if configured
            if aux_xy and (not hasattr(old_head, 'aux_xy') or getattr(old_head, 'aux_xy', None) is None):
                neck_ch = tuple(old_head.cv2[i][0].conv.in_channels for i in range(old_head.nl))
                old_head.aux_xy = nn.ModuleList(nn.Conv2d(c, 2, 1) for c in neck_ch)
                for layer in old_head.aux_xy:
                    nn.init.zeros_(layer.bias)
                    nn.init.zeros_(layer.weight)

            # Attach prediction-level self-attention on top-K scored predictions
            if prediction_refinement and (
                not hasattr(old_head, 'prediction_refinement_attn')
                or getattr(old_head, 'prediction_refinement_attn', None) is None
            ):
                from raycasted.model.blocks.head import PredictionRefinementAttention

                c3 = max(old_head.cv3[0][-1].in_channels, old_head.nc)
                old_head.prediction_refinement_attn = PredictionRefinementAttention(
                    feat_dim=c3,
                    num_heads=4,
                    ff_dim=256,
                )
                old_head.prediction_refinement_topk = prediction_refinement_topk
                if old_head._end2end_arg:
                    old_head.one2one_prediction_refinement_attn = copy.deepcopy(old_head.prediction_refinement_attn)
                    old_head.prediction_refinement_attn = None

            # --- Local window attention (o2o only, P2+P3) ---
            if local_window_attn and getattr(old_head, 'local_window_attn', None) is None:
                from raycasted.model.blocks.head import LocalWindowAttention

                c3 = old_head.cv3[0][-1].in_channels
                old_head.local_window_attn = nn.ModuleList(
                    [LocalWindowAttention(c3) for _ in range(2)]
                )

            # Rebuild cv2 with DCN if configured and not already present
            if dcn_in_reg_head:
                from ultralytics.nn.modules.conv import Conv as _Conv

                from raycasted.model.blocks.dcn import DCNConv

                has_dcn = any(isinstance(m, DCNConv) for m in old_head.cv2.modules())
                if not has_dcn:
                    c2 = old_head.cv2[0][1].conv.out_channels
                    neck_ch = tuple(old_head.cv2[i][0].conv.in_channels for i in range(old_head.nl))
                    from raycasted.model.blocks.head import RayRefinementBlock

                    block_cls = (
                        type(old_head.cv2[0][-2])
                        if not isinstance(old_head.cv2[0][-2], nn.Conv2d)
                        else RayRefinementBlock
                    )
                    old_head.cv2 = nn.ModuleList(
                        nn.Sequential(
                            _Conv(x, c2, 3),
                            DCNConv(c2, c2, 3),
                            block_cls(c2),
                            nn.Conv2d(c2, old_head.raycast_dim, 1),
                        )
                        for x in neck_ch
                    )
                    if old_head._end2end_arg:
                        old_head.one2one_cv2 = copy.deepcopy(old_head.cv2)

            # Rebuild cv3 with DCN if configured and not already present
            if dcn_in_cls_head:
                from ultralytics.nn.modules.conv import Conv as _Conv
                from ultralytics.nn.modules.conv import DWConv

                from raycasted.model.blocks.dcn import DCNConv

                has_dcn = any(isinstance(m, DCNConv) for m in old_head.cv3.modules())
                if not has_dcn:
                    c3 = old_head.cv3[0][-1].in_channels
                    neck_ch = tuple(old_head.cv2[i][0].conv.in_channels for i in range(old_head.nl))
                    old_head.cv3 = nn.ModuleList(
                        nn.Sequential(
                            nn.Sequential(DWConv(x, x, 3), _Conv(x, c3, 1)),
                            nn.Sequential(DWConv(c3, c3, 3), DCNConv(c3, c3, 3)),
                            nn.Conv2d(c3, old_head.nc, 1),
                        )
                        for x in neck_ch
                    )
                    if old_head._end2end_arg:
                        old_head.one2one_cv3 = copy.deepcopy(old_head.cv3)
        else:
            ch = _extract_neck_channels(old_head)
            nc = old_head.nc
            n_rays = _const.N_RAYS
            # Read head channel config from training_config (None → recommended defaults)
            tcfg = self.training_config
            head_channel_scale = tcfg.get('head_channel_scale', 0.5) if tcfg else 0.5
            head_channel_min = tcfg.get('head_channel_min', 64) if tcfg else 64
            refinement_kernel_size = tcfg.get('refinement_kernel_size', 3) if tcfg else 3
            new_head = RayCastDetect(
                nc=nc,
                end2end=True,
                ch=ch,
                n_rays=n_rays,
                head_channel_scale=head_channel_scale,
                head_channel_min=head_channel_min,
                cls_channel_scale=cls_channel_scale,
                cls_channel_min=cls_channel_min,
                refinement_kernel_size=refinement_kernel_size,
                aux_xy=aux_xy,
                prediction_refinement=prediction_refinement,
                prediction_refinement_topk=prediction_refinement_topk,
                inter_scale_competition=inter_scale_competition,
                inter_scale_temperature=inter_scale_temperature,
                local_competition=local_competition,
                local_competition_kernel=local_competition_kernel,
                local_competition_temperature=local_competition_temperature,
                dcn_in_reg_head=dcn_in_reg_head,
                dcn_in_cls_head=dcn_in_cls_head,
                hierarchical_cls=hierarchical_cls,
                hierarchical_cls_detach=hierarchical_cls_detach,
                hierarchical_binary_threshold=hierarchical_binary_threshold,
                local_window_attn=local_window_attn,
            )
            # Copy attributes set by parse_model (f=from layers, i=layer index, etc.)
            for attr in ('f', 'i', 'type'):
                if hasattr(old_head, attr):
                    setattr(new_head, attr, getattr(old_head, attr))
            # Copy stride from old head and initialise biases for new head
            new_head.stride = old_head.stride.clone()
            new_head.bias_init()
            model.model[-1] = new_head
            model.end2end = True

        # Patch init_criterion for FCN models only (RayCastDetect head).
        # RT-DETR and other decoder-based models have their own criterion
        # (Hungarian matching + VFL + ray L1 + polar IoU), not dual TAL.
        if isinstance(old_head, RayCastDetect):
            max_epochs = getattr(self.args, 'epochs', 200)
            model.init_criterion = _RayCastCriterionWrapper(
                model, max_epochs=max_epochs, training_config=self.training_config
            )

        # On resume, restore extra-head weights from checkpoint that were lost
        # when RayCastDetectionModel.__init__ reconstructed the model from YAML.
        # checkpoint_head_sd contains the full head state dict from the checkpoint,
        # including attention/quality head modules that the YAML doesn't specify.
        if checkpoint_head_sd:
            current_head = model.model[-1]
            current_sd = current_head.state_dict()
            restored_keys = []
            for k, v in checkpoint_head_sd.items():
                if k in current_sd and current_sd[k].shape == v.shape:
                    current_sd[k] = v
                    restored_keys.append(k)
            if restored_keys:
                current_head.load_state_dict(current_sd)
                n_restored = sum(checkpoint_head_sd[k].numel() for k in restored_keys)
                from ultralytics.utils import LOGGER

                LOGGER.info(f'Restored {len(restored_keys)} head keys from checkpoint ({n_restored:,} params)')

        # Load pretrained backbone weights if configured.
        # Only loads backbone layers (0-10), skipping neck/head so they
        # initialise randomly. This enables transfer learning from COCO
        # while allowing full freedom in neck/head architecture.
        bb_weights = (self.training_config or {}).get('pretrained_backbone')
        if bb_weights:
            _load_backbone_weights(model, bb_weights)

        return model

    def get_validator(self):
        """Return RayCastValidator for Shapely polygon mAP evaluation."""
        model_obj = self.model if hasattr(self, 'model') else None
        head = None
        if model_obj is not None:
            if hasattr(model_obj, 'model') and hasattr(model_obj.model, '__getitem__'):
                head = model_obj.model[-1]
            elif hasattr(model_obj, 'decode_head'):
                head = model_obj.decode_head

        if isinstance(head, (HybridRayCastDecoder,)):
            self.loss_names = ('ce_loss', 'centroid_loss', 'radial_loss')
        elif hasattr(model_obj, 'decode_head') and head is not None:
            from raycasted.model.blocks.lsp_detr_arch import LSPTransformer

            if isinstance(head, LSPTransformer):
                self.loss_names = ('ce_loss', 'centroid_loss', 'radial_loss')
        elif isinstance(head, RayCastRTDETRDecoder):
            self.loss_names = ('loss_class', 'loss_centroid', 'loss_ray')
        else:
            self.loss_names = (
                'xy_loss',
                'cls_loss',
                'l1_loss',
                'piou_loss',
                'smooth_loss',
                'aux_xy_loss',
                'quality_loss',
            )
        args_copy = copy.copy(self.args)
        if self.training_config and 'inference_conf' in self.training_config:
            args_copy.conf = self.training_config['inference_conf']
        return RayCastValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=args_copy,
            _callbacks=self.callbacks,
        )

    def build_dataset(self, img_path, mode='train', batch=None):  # noqa: ARG002
        """Build RayCastTileDataset from .npz tile files.

        Args:
            img_path: Path to directory containing .npz tiles.
            mode: 'train' or 'val'.
            batch: Unused (for API compat).

        Returns:
            RayCastTileDataset instance.
        """
        # Build augmentation config for the dataset
        augment_config = {}
        if mode == 'train' and self.training_config:
            augment_config = {
                k: v
                for k, v in self.training_config.items()
                if k
                in (
                    'stain_jitter',
                    'stain_hsv_h',
                    'stain_hsv_s',
                    'stain_hsv_v',
                    'stain_blur_prob',
                    'stain_blur_sigma',
                    'scale_augment',
                    'scale_range',
                    'translate_augment',
                    'translate_range',
                )
            }

        num_classes = None
        if hasattr(self, 'model') and hasattr(self.model, 'model'):
            head = self.model.model[-1]
            if hasattr(head, 'nc'):
                num_classes = head.nc

        return RayCastTileDataset(
            data_dir=img_path,
            crop_size=self.args.imgsz,
            augment=(mode == 'train'),
            augment_config=augment_config if augment_config else None,
            num_classes=num_classes,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Create DataLoader with raycast collate function.

        Args:
            dataset_path: Path to .npz tile directory.
            batch_size: Images per batch.
            rank: DDP process rank.
            mode: 'train' or 'val'.

        Returns:
            PyTorch DataLoader.
        """
        assert mode in {'train', 'val'}, f"Mode must be 'train' or 'val', not {mode}"
        dataset = self.build_dataset(dataset_path, mode, batch_size)

        # Validate data/model ray count match
        head = self.model.model[-1] if hasattr(self, 'model') and hasattr(self.model, 'model') else None
        if (
            head is not None
            and hasattr(head, 'n_rays')
            and hasattr(dataset, 'n_rays')
            and dataset.n_rays != head.n_rays
        ):
            raise ValueError(
                f'Data/model ray count mismatch: tiles have {dataset.n_rays} rays '
                f'but model head expects {head.n_rays}. '
                f'Re-ingest with --n-rays {head.n_rays} or train with --n-rays {dataset.n_rays}.'
            )
        use_weighted = (
            mode == 'train'
            and (self.training_config or {}).get('weighted_sampling', False)
            and dataset.num_classes is not None
        )

        sampler = None
        shuffle = False
        if use_weighted:
            gamma = self.training_config.get('sampler_gamma', 0.85)
            sampler = dataset.get_sampler(gamma=gamma)
        else:
            shuffle = mode == 'train'

        # Propagate steps_per_epoch to the loss criterion so that the o2m/o2o
        # decay schedule and epoch-dependent annealing use the correct rate.
        # Only set on the train dataloader call (first call with mode='train').
        if mode == 'train' and hasattr(self, 'model'):
            criterion = getattr(self.model, 'criterion', None)
            if criterion is None and hasattr(self.model, 'init_criterion'):
                ic = self.model.init_criterion
                if isinstance(ic, _RayCastCriterionWrapper):
                    ic._steps_per_epoch = max(1, len(dataset) // max(batch_size, 1))

        dl = InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.workers,
            collate_fn=_raycast_collate_fn,
            drop_last=self.args.compile and mode == 'train',
            multiprocessing_context='forkserver' if self.args.workers > 0 else None,
        )

        # After the dataloader is built, set steps_per_epoch on the live
        # criterion (created by resume_training or _setup_train).  The
        # criterion may not exist yet on the first get_dataloader(train) call
        # (it's created later in resume_training), so we also update the
        # wrapper above for the deferred case.
        if mode == 'train' and hasattr(self, 'model'):
            unwrapped = self.model
            if hasattr(unwrapped, 'module'):
                unwrapped = unwrapped.module
            criterion = getattr(unwrapped, 'criterion', None)
            if criterion is not None and hasattr(criterion, 'set_steps_per_epoch'):
                spe = max(1, math.ceil(len(dataset) / max(batch_size, 1)))
                criterion.set_steps_per_epoch(spe)

        return dl

    def preprocess_batch(self, batch):
        """Move batch tensors to device.

        Images are already float32 normalised by RayCastTileDataset,
        so no /255 division is needed.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == 'cuda')
        return batch

    def setup_model(self):
        """Patch old LSP-DETR checkpoints before Ultralytics setup_model."""
        ckpt_path = self.model if isinstance(self.model, str) else None
        from raycasted.model.lsp_detr_model import LSPDetrDetectionModel

        if ckpt_path and ckpt_path.endswith('.pt'):
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
                if isinstance(model, LSPDetrDetectionModel):
                    if not hasattr(model, 'yaml'):
                        model.yaml = {'nc': model.nc, 'head': [[[2, 4, 8], 1, 'LSPDetrModel', ['nc', 64]]]}
                    if not hasattr(model, 'end2end'):
                        model.end2end = False
                    torch.save(ckpt, ckpt_path)
            except Exception:
                pass
        return super().setup_model()

    def set_model_attributes(self):
        """Set model attributes and training metadata.

        Replicates core logic from parent (nc, names, args) without
        calling set_head_attr which doesn't exist. Adds training_args
        dict needed by RayCastPredictor and ONNX export.
        """
        model = self.model
        if hasattr(model, 'decode_head'):
            head = model.decode_head
        else:
            head = model.model[-1]

        self.model.nc = self.data.get('nc', getattr(self.model, 'nc', head.nc))
        assert head.nc == self.model.nc, (
            f'Head nc ({head.nc}) != data nc ({self.model.nc}). '
            f'Model YAML likely has wrong nc. Fix the YAML or pass nc to get_model().'
        )
        self.model.names = self.data.get('names', getattr(self.model, 'names', {}))
        self.model.args = self.args

        # Training metadata for inference and export
        n_rays = head.n_rays if hasattr(head, 'n_rays') else _const.N_RAYS
        strides = head.stride.tolist() if hasattr(head, 'stride') and head.stride.sum() > 0 else [4, 8, 16]
        self.model.training_args = {
            'crop_size': self.args.imgsz,
            'imgsz': self.args.imgsz,
            'nc': head.nc,
            'n_rays': n_rays,
            'strides': strides,
        }

        # Re-initialise biases with correct crop_size (Ultralytics calls bias_init
        # during model construction with no access to training config)
        if hasattr(head, 'bias_init') and not isinstance(head, RayCastRTDETRDecoder):
            tcfg = self.training_config or {}
            head.bias_init(
                crop_size=self.args.imgsz,
                native_mpp=tcfg.get('native_mpp', 0.25),
                min_nucleus_diameter_um=tcfg.get('min_nucleus_diameter_um', 3.5),
            )


def train(**kwargs):
    """Train a RayCastED polygon detection model.

    Convenience function that creates a RayCastTrainer and runs training.

    Args:
        **kwargs: Override arguments passed to RayCastTrainer.
            Key args: model, data, epochs, batch, imgsz.

    Returns:
        RayCastTrainer instance after training completes.
    """
    overrides = {
        'mosaic': 0.0,
        'mixup': 0.0,
        **kwargs,
    }
    trainer = RayCastTrainer(overrides=overrides)
    trainer.train()
    return trainer


if __name__ == '__main__':
    train()
