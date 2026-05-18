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
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import initialize_weights

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.utils import constants as _const
from raycasted.model.blocks.head import RayCastDetect
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

    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': targets[:, 2:],  # [N, 2+n_rays] = cx, cy, d_1..d_n
        'ori_shape': torch.stack(ori_shapes),
        'ratio_pad': ratio_pads,
        'im_file': im_files,
    }


class _RayCastCriterionWrapper:
    """Picklable callable that wraps RayCastE2ELoss creation.

    Replaces a lambda so that torch.save / torch.load can serialize the model.
    Called as: model.init_criterion() — the wrapper is callable.
    """

    def __init__(self, model, max_epochs: int = 200, training_config: dict | None = None):
        self._model = model
        self._max_epochs = max_epochs
        self._training_config = training_config

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

        cw = tcfg.get('class_weights', None)
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
            plb_enabled=tcfg.get('plb_enabled', False),
            bg_cls_decay=tcfg.get('bg_cls_decay', 1.0),
            fg_cls_boost=tcfg.get('fg_cls_boost', 0.0),
            soft_targets=tcfg.get('soft_targets', False),
            class_weights=self._build_class_weights(tcfg),
            o2o_topk2_start=tcfg.get('o2o_topk2_start', 1),
            o2o_topk2_anneal_epoch=tcfg.get('o2o_topk2_anneal_epoch', 0),
            hungarian_phase2_start=tcfg.get('hungarian_phase2_start', 0),
            hungarian_phase3_start=tcfg.get('hungarian_phase3_start', 0),
            hungarian_max_weight=tcfg.get('hungarian_max_weight', 0.9),
            hungarian_cost_class=tcfg.get('hungarian_cost_class', 1.0),
            hungarian_cost_centroid=tcfg.get('hungarian_cost_centroid', 1.0),
            hungarian_cost_ray=tcfg.get('hungarian_cost_ray', 1.0),
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

        # Register GradNorm callback — updates dynamic loss weights after each step
        self.add_callback('on_train_batch_end', _gradnorm_update_callback)
        # Register best-epoch logger — prints after each validation epoch
        self.add_callback('on_fit_epoch_end', _best_epoch_callback)
        # Register per-epoch LR logging — critical for MuSGD multi-group debugging
        self.add_callback('on_fit_epoch_end', _lr_log_callback)

    def plot_training_labels(self):
        """Skip standard bbox label plotting — incompatible with raycast polygon data."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create YOLO model with RayCastDetect head and RayCastE2ELoss.

        Uses RayCastDetectionModel (custom builder) instead of standard
        DetectionModel. Replaces the Detect head with RayCastDetect and
        patches init_criterion to return RayCastE2ELoss.

        Args:
            cfg: Model config path or YAML name.
            weights: Pretrained weights path.
            verbose: Print model info.

        Returns:
            RayCastDetectionModel with RayCastDetect head.
        """
        register_raycast_head()
        nc = self.data.get('nc') if hasattr(self, 'data') and self.data else None
        model = RayCastDetectionModel(cfg, ch=3, nc=nc, verbose=verbose)

        # Replace Detect head → RayCastDetect
        old_head = model.model[-1]
        tcfg = self.training_config
        aux_xy = bool(tcfg.get('aux_xy_weight', 0) > 0) if tcfg else False

        if isinstance(old_head, RayCastDetect):
            # YAML already specifies RayCastDetect — ensure n_rays matches
            # the configured value (YAML args don't include n_rays, so it
            # may have been created with the wrong default).
            if old_head.n_rays != _const.N_RAYS:
                old_head.n_rays = _const.N_RAYS
                old_head.raycast_dim = 2 + _const.N_RAYS
                old_head.no = old_head.nc + old_head.raycast_dim

            # Attach auxiliary xy head if configured
            if aux_xy and (not hasattr(old_head, 'aux_xy') or getattr(old_head, 'aux_xy', None) is None):
                neck_ch = tuple(old_head.cv2[i][0].conv.in_channels for i in range(old_head.nl))
                old_head.aux_xy = nn.ModuleList(nn.Conv2d(c, 2, 1) for c in neck_ch)
                for layer in old_head.aux_xy:
                    nn.init.zeros_(layer.bias)
                    nn.init.zeros_(layer.weight)
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
                refinement_kernel_size=refinement_kernel_size,
                aux_xy=aux_xy,
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
        self.loss_names = ('xy_loss', 'cls_loss', 'l1_loss', 'piou_loss', 'smooth_loss', 'aux_xy_loss')
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

        return InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.workers,
            collate_fn=_raycast_collate_fn,
            drop_last=self.args.compile and mode == 'train',
            multiprocessing_context='forkserver' if self.args.workers > 0 else None,
        )

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
        strides = head.stride.tolist() if hasattr(head, 'stride') and head.stride.sum() > 0 else [8, 16, 32]
        self.model.training_args = {
            'crop_size': self.args.imgsz,
            'imgsz': self.args.imgsz,
            'nc': head.nc,
            'n_rays': head.n_rays,
            'strides': strides,
        }

        # Re-initialise biases with correct crop_size (Ultralytics calls bias_init
        # during model construction with no access to training config)
        if hasattr(head, 'bias_init'):
            head.bias_init(crop_size=self.args.imgsz)


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
