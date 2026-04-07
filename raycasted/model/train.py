"""RayCastED — Training Pipeline (Phase 10).

Wires all model components into the Ultralytics training loop:
  - RayCastDetect head (replaces standard Detect)
  - RayCastE2ELoss (5-term polygon loss with smoothness annealing)
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

import numpy as np
import torch
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.model.head import RayCastDetect
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


def _raycast_collate_fn(batch: list) -> dict:
    """Collate (image, labels) tuples into a batch dict for training.

    Produces the dict format expected by RayCastDetectionLoss:
        'img':        [B, 3, H, W] float32
        'batch_idx':  [sum_M] float32
        'cls':        [sum_M] float32 (class id)
        'bboxes':     [sum_M, 34] float32 (cx, cy, d_1..d_32)

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
        targets = torch.zeros((0, 36), dtype=torch.float32)

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
        'bboxes': targets[:, 2:],  # [N, 34] = cx, cy, d_1..d_32
        'ori_shape': torch.stack(ori_shapes),
        'ratio_pad': ratio_pads,
        'im_file': im_files,
    }


class _RayCastCriterionWrapper:
    """Picklable callable that wraps RayCastE2ELoss creation.

    Replaces a lambda so that torch.save / torch.load can serialize the model.
    Called as: model.init_criterion() — the wrapper is callable.
    """

    def __init__(self, model):
        self._model = model

    def __call__(self):
        return RayCastE2ELoss(self._model)


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

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialise RayCastTrainer with mosaic/mixup forced off.

        Args:
            cfg: Default configuration.
            overrides: Dict of parameter overrides.
            _callbacks: Callback functions.
        """
        overrides = overrides or {}
        super().__init__(cfg, overrides, _callbacks)
        # Force mosaic/mixup off — they corrupt polygon targets (GAP-06)
        self.args.mosaic = 0.0
        self.args.mixup = 0.0

    def plot_training_labels(self):
        """Skip standard bbox label plotting — incompatible with 34-dim polygon data."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create YOLO model with RayCastDetect head and RayCastE2ELoss.

        Loads a standard YOLO architecture, then replaces the Detect head
        with RayCastDetect and patches init_criterion to return
        RayCastE2ELoss (handles both initial creation and resume).

        Args:
            cfg: Model config path or YAML name.
            weights: Pretrained weights path.
            verbose: Print model info.

        Returns:
            DetectionModel with RayCastDetect head.
        """
        register_raycast_head()
        model = super().get_model(cfg, weights, verbose)

        # Replace Detect head → RayCastDetect
        old_head = model.model[-1]
        if not isinstance(old_head, RayCastDetect):
            ch = _extract_neck_channels(old_head)
            nc = old_head.nc
            new_head = RayCastDetect(nc=nc, end2end=True, ch=ch)
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
        model.init_criterion = _RayCastCriterionWrapper(model)

        return model

    def get_validator(self):
        """Return RayCastValidator for Shapely polygon mAP evaluation."""
        self.loss_names = ('xy_loss', 'cls_loss', 'l1_loss', 'piou_loss', 'smooth_loss')
        return RayCastValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy.copy(self.args),
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
        return RayCastTileDataset(
            data_dir=img_path,
            crop_size=self.args.imgsz,
            augment=(mode == 'train'),
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
        return InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=self.args.workers,
            collate_fn=_raycast_collate_fn,
            drop_last=self.args.compile and mode == 'train',
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
        self.model.names = self.data.get('names', getattr(self.model, 'names', {}))
        self.model.args = self.args

        # Training metadata for inference and export
        strides = head.stride.tolist() if hasattr(head, 'stride') and head.stride.sum() > 0 else [8, 16, 32]
        self.model.training_args = {
            'crop_size': self.args.imgsz,
            'imgsz': self.args.imgsz,
            'nc': head.nc,
            'n_rays': 32,
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
