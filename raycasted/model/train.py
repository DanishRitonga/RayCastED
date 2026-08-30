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
import re
from copy import deepcopy
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.optim import MuSGD
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import initialize_weights, unwrap_model

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.utils import constants as _const
from raycasted.model.blocks.head import RayCastDetect
from raycasted.model.builder import raycasted_parse_model
from raycasted.model.loss import RayCastE2ELoss
from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel
from raycasted.model.nulite_model import NuLiteRayCastModel
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

    # NP mask + seed map: rasterize binary nuclei mask from normalized polygon labels.
    # Labels are normalized [0,1]; multiply cx/cy/rays by crop_size to get pixel coords.
    # seed_map = InstanSeg-style target (instance_wise_edt): per-instance
    # distance-to-boundary, PER-INSTANCE normalized (each nucleus center = 1,
    # boundary = 0), then transformed to (edt - 0.5) * 15 => range [-7.5, +7.5]
    # (background negative, center positive). Matches InstanSeg exactly so the
    # NP head can be queried by local-maxima for DETR query selection.
    np_masks = []
    seed_maps = []
    for labels in labels_list:
        h = w = images.shape[-1]
        mask = np.zeros((h, w), dtype=np.float32)
        inst = np.zeros((h, w), dtype=np.int32)
        if labels.shape[0] > 0:
            n_rays = labels.shape[1] - 3
            angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
            cos_a = np.cos(angles)
            sin_a = np.sin(angles)
            for idx, ann in enumerate(labels):
                cx = float(ann[1]) * w
                cy = float(ann[2]) * h
                rays = ann[3:] * w
                xs = cx + rays * cos_a
                ys = cy + rays * sin_a
                polygon = np.stack([xs, ys], axis=1).astype(np.int32)
                cv2.fillPoly(mask, [polygon], 1.0)
                cv2.fillPoly(inst, [polygon], idx + 1)
        np_masks.append(torch.from_numpy(mask[None]))  # (1, H, W)
        # Per-instance normalized EDT: center=1 regardless of nucleus size.
        dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 3)
        seed = np.zeros_like(dist)
        for label in range(1, int(inst.max()) + 1):
            m = inst == label
            if m.sum() == 0:
                continue
            dmax = float(dist[m].max())
            if dmax > 0:
                seed[m] = dist[m] / dmax
        # InstanSeg transform: symmetric around 0, scaled to mimic CE range.
        seed = (seed - 0.5) * 15.0
        seed_maps.append(torch.from_numpy(seed[None]))  # (1, H, W)

    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': targets[:, 2:],  # [N, 2+n_rays] = cx, cy, d_1..d_n
        'np_mask': torch.stack(np_masks),  # (B, 1, H, W)
        'seed_map': torch.stack(seed_maps),  # (B, 1, H, W) normalized distance-to-boundary
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

    def __call__(self):
        tcfg = self._training_config or {}
        return RayCastE2ELoss(
            self._model,
            max_epochs=self._max_epochs,
            tal_topk=tcfg.get('tal_topk', 13),
            assigner_radius_scale=tcfg.get('assigner_radius_scale', 1.5),
            assigner_alpha=tcfg.get('assigner_alpha', 0.5),
            assigner_beta=tcfg.get('assigner_beta', 6.0),
            log_ray_loss=tcfg.get('log_ray_loss', False),
            focal_gamma=tcfg.get('focal_gamma', 0.0),
            focal_alpha=tcfg.get('focal_alpha', 0.25),
            align_threshold=tcfg.get('align_threshold', 0.0),
            lambda_aux_xy=tcfg.get('aux_xy_weight', 0.0),
            aux_xy_ramp_epochs=tcfg.get('aux_xy_ramp_epochs', 100),
            bg_fg_ratio=tcfg.get('bg_fg_ratio', 3),
            plb_enabled=tcfg.get('plb_enabled', False),
            focal_gamma_o2o=tcfg.get('focal_gamma_o2o', None),
            focal_alpha_o2o=tcfg.get('focal_alpha_o2o', None),
            bg_fg_ratio_o2o=tcfg.get('bg_fg_ratio_o2o', None),
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
            steps_per_epoch=self._steps_per_epoch,
            hierarchical_cls=tcfg.get('hierarchical_cls', False),
            nc_override=tcfg.get('nc_override', None),
            cls_only_tal=tcfg.get('cls_only_tal', False),
            o2o_distill_weight=tcfg.get('o2o_distill_weight', 0.0),
        )


class RayCastDetectionModel(DetectionModel):
    """Custom DetectionModel using raycasted_parse_model builder.

    Replaces ultralytics' parse_model() with our custom builder that
    supports custom blocks (ResoConv) without monkey-patching.

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

        # Register best-epoch logger — prints after each validation epoch
        self.add_callback('on_fit_epoch_end', _best_epoch_callback)
        # Register per-epoch LR logging — critical for MuSGD multi-group debugging
        self.add_callback('on_fit_epoch_end', _lr_log_callback)

    def plot_training_labels(self):
        """Skip standard bbox label plotting — incompatible with raycast polygon data."""

    def plot_training_samples(self, batch, ni):
        """Skip training-batch mosaic plotting — raycast polygons (66-dim) break xywh2xyxy."""

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer with a MuSGD routing fix for 3D params.

        Ultralytics routes every `param.ndim >= 2` tensor to the Muon group, but
        `muon_update` only reshapes 4D conv filters to 2D before
        `zeropower_via_newtonschulz5` (which asserts `len(G.shape) == 2`). FastViT's
        `layer_scale.gamma` tensors are (C, 1, 1) 3D, so they crash MuSGD. Route
        3D params to the no-decay group (semantically correct for per-channel
        scale factors) and keep only 2D/4D tensors in the Muon group.
        """
        g = [{}, {}, {}, {}]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers
        if name == 'auto':
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get('nc', 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ('MuSGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0

        use_muon = name == 'MuSGD'
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if use_muon and param.ndim in (2, 4):
                    g[3][fullname] = param  # muon params (2D matrices / 4D conv filters)
                elif 'bias' in fullname:  # bias (no decay)
                    g[2][fullname] = param
                elif isinstance(module, bn) or 'logit_scale' in fullname or param.ndim == 3:
                    g[1][fullname] = param  # norm weight / layer_scale (no decay)
                else:  # weight (with decay)
                    g[0][fullname] = param
        if not use_muon:
            g = [x.values() for x in g[:3]]

        optimizers = {'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'SGD', 'MuSGD', 'auto'}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == 'SGD' or name == 'MuSGD':
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                'Request support for addition optimizers at https://github.com/ultralytics/ultralytics.'
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]
        g[2] = {'params': g[2], **optim_args, 'param_group': 'bias'}
        g[0] = {'params': g[0], **optim_args, 'weight_decay': decay, 'param_group': 'weight'}
        g[1] = {'params': g[1], **optim_args, 'weight_decay': 0.0, 'param_group': 'bn'}
        muon, sgd = (0.2, 1.0)
        if use_muon:
            num_params[0] = len(g[3])
            g[3] = {'params': g[3], **optim_args, 'weight_decay': decay, 'use_muon': True, 'param_group': 'muon'}
            # higher lr for certain parameters in MuSGD when finetuning
            pattern = re.compile(r'(?=.*23)(?=.*cv3)|proto\.semseg')
            g_ = []  # new param groups
            for x in g:
                p = x.pop('params')
                p1 = [v for k, v in p.items() if pattern.search(k)]
                p2 = [v for k, v in p.items() if not pattern.search(k)]
                g_.extend([{'params': p1, **x, 'lr': lr * 3}, {'params': p2, **x}])
            g = g_
        optimizer = getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=g)

        LOGGER.info(
            f'{colorstr("optimizer:")} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups '
            f'{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)'
        )
        return optimizer

    def validate(self):
        """Mask fitness during the early warmup window (fluke-spike guard).

        DETR-style runs have bimodal early metrics: a lucky epoch where the
        conf distribution shifts past inference_conf produces a fitness spike
        (e.g. ep2 bPQ 0.235 with recall 0.03) that poisons best-epoch tracking
        and starts the EarlyStopping patience clock. Before
        ``fitness_warmup_epochs`` we run the validator directly (NOT through
        ``super().validate()``, which updates ``best_fitness`` with the real
        fitness before we could mask it) and return -inf fitness, keeping
        best_fitness/best.pt honest.
        """
        warmup = (self.training_config or {}).get('fitness_warmup_epochs', 0)
        if warmup and self.epoch < warmup:
            metrics = self.validator(self)
            if metrics is None:
                return None, None
            # Drop 'fitness' so the CSV header/rows stay consistent: the parent
            # validate() also pops it, and re-adding it here (the old behaviour)
            # made the warmup rows carry an extra column -> column misalignment.
            metrics.pop('fitness', None)
            return metrics, float('-inf')
        return super().validate()

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

    def _get_nulite_detr_model(self, nc=None, weights=None, verbose=True):
        """Build the NuLite DETR (single query-based head) model.

        FastViT encoder + NuLite decoder feed a single DETR head whose queries
        are seeded from local maxima of the NP seed map (InstanSeg-style
        self-prompting). No PAN/conditioners/multi-scale FCN heads. Uses the
        NuLiteDETRLoss (Hungarian 1:1 + VFL).
        """
        tcfg = self.training_config or {}
        nc = nc if nc is not None else 5
        is_resume = weights is not None and isinstance(weights, torch.nn.Module)

        model = NuLiteRayCastDETRModel(
            nc=nc,
            n_rays=_const.N_RAYS,
            variant=tcfg.get('nulite_variant', 'fastvit_s12'),
            pretrained=bool(tcfg.get('pretrained', True)) and not is_resume,
            lambda_seg=float(tcfg.get('lambda_seg', 1.0)),
            nq=int(tcfg.get('detr_nq', 300)),
            ndl=int(tcfg.get('detr_ndl', 3)),
            hd=int(tcfg.get('detr_hd', 256)),
            d_ffn=int(tcfg.get('detr_d_ffn', 1024)),
            seed_threshold=float(tcfg.get('detr_seed_threshold', 0.5)),
            peak_distance=int(tcfg.get('detr_peak_distance', 3)),
            grid_size=float(tcfg.get('detr_grid_size', 0.05)),
            query_selection=tcfg.get('detr_query_selection', 'grid'),
            seed_feature=bool(tcfg.get('detr_seed_feature', True)),
            no_object=bool(tcfg.get('detr_no_object', True)),
            seed_in_content=bool(tcfg.get('detr_seed_in_content', True)),
            no_object_weight=float(tcfg.get('detr_no_object_weight', 1.0)),
            mds=bool(tcfg.get('detr_mds', True)),
            cost_inside=float(tcfg.get('detr_cost_inside', 10.0)),
            seed_map_target=bool(tcfg.get('seed_map_target', True)),
            use_decoder=bool(tcfg.get('detr_use_decoder', True)),
            shared_layers=bool(tcfg.get('detr_shared_layers', False)),
            backbone=tcfg.get('detr_backbone', 'nulite'),
            use_seed=bool(tcfg.get('detr_use_seed', True)),
            yolo11_scale=tcfg.get('yolo11_scale', 'n'),
            yolo11_ckpt=tcfg.get('yolo11_ckpt'),
            verbose=verbose,
        )

        # On resume, restore full checkpoint weights.
        if is_resume:
            model.load_state_dict(weights.state_dict())

        nulite_ckpt = tcfg.get('nulite_ckpt')
        if nulite_ckpt:
            model.load_nulite_ckpt(nulite_ckpt)

        if tcfg.get('freeze_encoder_decoder', False):
            model.freeze_encoder_decoder_np()

        return model

    def _get_nulite_model(self, nc=None, weights=None, verbose=True):
        """Build the NuLite (FastViT + NuLite decoder + NP seg head) model.

        FastViT returns multi-scale encoder features that the YAML builder
        cannot route, so the model is built in pure Python and reuses the
        v1 RayCastDetect head + RayCastE2ELoss + ETL pipeline unchanged.
        """
        tcfg = self.training_config or {}
        nc = nc if nc is not None else 5
        is_resume = weights is not None and isinstance(weights, torch.nn.Module)

        model = NuLiteRayCastModel(
            nc=nc,
            n_rays=_const.N_RAYS,
            variant=tcfg.get('nulite_variant', 'fastvit_s12'),
            pretrained=bool(tcfg.get('pretrained', True)) and not is_resume,
            lambda_seg=float(tcfg.get('lambda_seg', 1.0)),
            seed_map_target=bool(tcfg.get('seed_map_target', False)),
            gate_scale=float(tcfg.get('gate_scale', 1.0)),
            head_channel_scale=tcfg.get('head_channel_scale', 0.5),
            head_channel_min=tcfg.get('head_channel_min', 64),
            cls_channel_scale=tcfg.get('cls_channel_scale', 1.0),
            cls_channel_min=tcfg.get('cls_channel_min', 0),
            aux_xy=bool(tcfg.get('aux_xy_weight', 0) > 0),
            dcn_in_reg_head=bool(tcfg.get('dcn_in_reg_head', False)),
            dcn_in_cls_head=bool(tcfg.get('dcn_in_cls_head', False)),
            hierarchical_cls=bool(tcfg.get('hierarchical_cls', False)),
            hierarchical_cls_detach=bool(tcfg.get('hierarchical_cls_detach', True)),
            hierarchical_binary_threshold=float(tcfg.get('hierarchical_binary_threshold', 0.01)),
            verbose=verbose,
        )

        # On resume, restore full checkpoint weights (backbone + decoder + head).
        if is_resume:
            model.load_state_dict(weights.state_dict())

        model.init_criterion = _RayCastCriterionWrapper(
            model,
            max_epochs=getattr(self.args, 'epochs', 200),
            training_config=self.training_config,
        )

        nulite_ckpt = tcfg.get('nulite_ckpt')
        if nulite_ckpt:
            model.load_nulite_ckpt(nulite_ckpt)

        # Phase-2a freeze: freeze encoder+decoder+NP, train head+conditioners only.
        if tcfg.get('freeze_encoder_decoder', False):
            model.freeze_encoder_decoder_np()

        return model

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create YOLO model with RayCastDetect head and RayCastE2ELoss.

        Uses RayCastDetectionModel (custom builder) instead of standard
        DetectionModel. Replaces the Detect head with RayCastDetect and
        patches init_criterion to return RayCastE2ELoss.

        Args:
            cfg: Model config path or YAML name.
            weights: Pretrained weights path or checkpoint model object (on resume).
            verbose: Print model info.

        Returns:
            RayCastDetectionModel.
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

        _const.configure_rays(self.training_config.get('n_rays', 64) if self.training_config else 64)

        if tcfg.get('architecture') == 'nulite':
            return self._get_nulite_model(nc=nc, weights=weights, verbose=verbose)

        if tcfg.get('architecture') == 'nulite_detr':
            return self._get_nulite_detr_model(nc=nc, weights=weights, verbose=verbose)

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
        inter_scale_competition = bool(tcfg.get('inter_scale_competition', False)) if tcfg else False
        inter_scale_temperature = tcfg.get('inter_scale_temperature', 1.0) if tcfg else 1.0
        cls_channel_scale = tcfg.get('cls_channel_scale', 1.0) if tcfg else 1.0
        cls_channel_min = tcfg.get('cls_channel_min', 0) if tcfg else 0
        dcn_in_reg_head = bool(tcfg.get('dcn_in_reg_head', False)) if tcfg else False
        dcn_in_cls_head = bool(tcfg.get('dcn_in_cls_head', False)) if tcfg else False
        hierarchical_cls = bool(tcfg.get('hierarchical_cls', False)) if tcfg else False
        hierarchical_cls_detach = bool(tcfg.get('hierarchical_cls_detach', True)) if tcfg else True
        hierarchical_binary_threshold = float(tcfg.get('hierarchical_binary_threshold', 0.01)) if tcfg else 0.01

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

            # Attach auxiliary xy head if configured
            if aux_xy and (not hasattr(old_head, 'aux_xy') or getattr(old_head, 'aux_xy', None) is None):
                neck_ch = tuple(old_head.cv2[i][0].conv.in_channels for i in range(old_head.nl))
                old_head.aux_xy = nn.ModuleList(nn.Conv2d(c, 2, 1) for c in neck_ch)
                for layer in old_head.aux_xy:
                    nn.init.zeros_(layer.bias)
                    nn.init.zeros_(layer.weight)

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
            new_head = RayCastDetect(
                nc=nc,
                end2end=True,
                ch=ch,
                n_rays=n_rays,
                head_channel_scale=head_channel_scale,
                head_channel_min=head_channel_min,
                cls_channel_scale=cls_channel_scale,
                cls_channel_min=cls_channel_min,
                aux_xy=aux_xy,
                inter_scale_competition=inter_scale_competition,
                inter_scale_temperature=inter_scale_temperature,
                dcn_in_reg_head=dcn_in_reg_head,
                dcn_in_cls_head=dcn_in_cls_head,
                hierarchical_cls=hierarchical_cls,
                hierarchical_cls_detach=hierarchical_cls_detach,
                hierarchical_binary_threshold=hierarchical_binary_threshold,
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

        # Patch init_criterion for both initial creation and resume path.
        # The resume path calls model.init_criterion() to re-create the loss,
        # so monkey-patching ensures RayCastE2ELoss is always used.
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
        if (self.training_config or {}).get('architecture') == 'nulite_detr':
            self.loss_names = ('cls_loss', 'xy_loss', 'l1_loss', 'piou_loss', 'smooth_loss')
        else:
            self.loss_names = (
                'xy_loss',
                'cls_loss',
                'l1_loss',
                'piou_loss',
                'smooth_loss',
                'distill_loss',
                'aux_xy_loss',
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

    def set_model_attributes(self):
        """Set model attributes and training metadata.

        Replicates core logic from parent (nc, names, args) without
        calling set_head_attr which doesn't exist. Adds training_args
        dict needed by RayCastPredictor and ONNX export.
        """
        head = self.model.model[-1]
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
        if hasattr(head, 'bias_init'):
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
