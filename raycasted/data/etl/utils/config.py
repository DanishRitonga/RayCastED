from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# === NESTED MODELS FOR COMPLEX OBJECTS ===
class SplitArgs(BaseModel):
    regex: str | None = None
    split_map: dict[str, str] | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


class ModalityDirs(BaseModel):
    image_dir: str
    mask_dir: str


class ModalityPairingRule(BaseModel):
    match_extension: str
    suffix_to_replace: str | None = None
    add_suffix: str | None = None


class CSVColumnMap(BaseModel):
    x_coords: str | None = None
    y_coords: str | None = None
    category: str | None = None


# === TRAINING SETTINGS ===
class TrainingSettings(BaseModel):
    """Configurable training parameters for RayCastED.

    All defaults are recommended/improved values. When the ``training``
    section is absent from the YAML config, legacy constants are used
    instead (see pipeline.py).
    """

    # Head architecture
    head_channel_scale: float = 0.5  # c2 = max(head_channel_min, ch * scale). Was 0.25
    head_channel_min: int = 64  # minimum channels in polygon head. Was 16
    cls_channel_scale: float = 1.0  # c3 = max(cls_channel_min, ch[0] * scale). 1.0 = original
    cls_channel_min: int = 0  # minimum cls head intermediate channels. 0 = use ch[0]

    # LSP-DETR ray initialization: start at minimum plausible nucleus radius
    # so gradient is unidirectional (expand only), avoiding conflicting shrink/expand signals
    native_mpp: float = 0.25  # microns per pixel (PanNuke 40x = 0.25 um/px)
    min_nucleus_diameter_um: float = 3.5  # minimum nucleus diameter (PMC 4600468), radius = 1.75um → 7px

    # Learning rate
    cos_lr: bool = True  # cosine LR schedule. Was False
    warm_restarts: bool = False  # CosineAnnealingWarmRestarts instead of single cosine
    warm_restarts_T0: int = 200  # noqa: N815 — matches PyTorch API. First restart period (epochs)
    warm_restarts_T_mult: int = 2  # noqa: N815 — matches PyTorch API. Period multiplier after each restart
    warm_restarts_eta_min: float = 0.0001  # minimum LR at cycle bottom

    # Assigner
    assigner_radius_scale: float = 2.0  # containment radius multiplier. Was 1.5

    # Classification loss — Focal or BCE
    # Previous "focal tested, harmful" result was caused by alpha=1.0 which
    # zeroed all background gradients. Standard alpha=0.25 works correctly.
    focal_gamma: float = 0.0  # 0 = BCE, >0 = Focal loss (recommended: 2.0)
    focal_alpha: float = 0.25  # positive weight (standard: 0.25)
    log_ray_loss: bool = False  # log-space L1 on rays for scale-invariant errors
    bg_fg_ratio: int = 3  # max bg anchors per fg anchor in cls loss
    ohem_bg_ratio: float = 0.0  # OHEM: keep hardest K bg per fg (0.0=random, >0=hard example mining)

    # Pixel-Level Balancing
    plb_enabled: bool = False  # area-based fg weighting to boost small nuclei
    bg_cls_decay: float = 1.0  # background classification loss decay (1.0=full, 0.0=off)
    fg_cls_boost: float = 0.0  # fg cls boost by alignment quality (0.0=disabled)
    fg_cls_quality_scale: float = 0.0  # multiplicative quality re-weight (0.0=disabled, 1.0=full quality scaling)

    # Soft targets — keep assigner quality scores instead of hard 0/1 binarisation
    soft_targets: bool = False
    soft_targets_o2o: bool | None = None  # per-branch override for o2o

    # Per-o2o classification loss overrides — the o2o branch must be a
    # "background specialist" because NMS-free inference has no safety net.
    # Set to None to inherit the base value.
    focal_gamma_o2o: float | None = None
    focal_alpha_o2o: float | None = None
    bg_fg_ratio_o2o: int | None = None
    ohem_bg_ratio_o2o: float | None = None  # OHEM for o2o branch (None=inherit base)
    bg_cls_decay_o2o: float | None = None  # per-branch bg suppression for o2o
    fg_cls_boost_o2o: float | None = None  # per-branch quality re-weighting for o2o
    fg_cls_quality_scale_o2o: float | None = None  # per-branch multiplicative quality for o2o

    # Loss weights (literature: regression should be 3-7x higher than cls)
    lambda_l1: float = 14.0  # L1 ray distance weight
    lambda_piou: float = 13.0  # Polar IoU weight
    lambda_cls: float = 2.0  # classification weight
    lambda_xy: float = 500.0  # centroid xy weight

    # Per-class inverse-frequency weights (sqrt-smoothed). None = no weighting.
    class_weights: list[float] | None = None

    # Alignment threshold
    align_threshold: float = 0.0

    # o2o topk2 annealing (one-to-few → strict 1:1)
    o2o_topk2_start: int = 1
    o2o_topk2_anneal_epoch: int = 0
    o2o_topk2_anneal_end: float = 0.5  # fraction of max_epochs where topk2 reaches 1 (0.5 = 50%)

    # Sigma annealing: broad→tight assignment over training (DCFL, CVPR 2023)
    sigma_anneal_start: float = 0.0  # initial radius_scale (0 = use assigner_radius_scale)
    sigma_anneal_end: float = 0.0  # final radius_scale (0 = no annealing)
    sigma_anneal_epoch: int = 0  # epoch to start annealing (0 = disabled)

    # STAL: Small-Target-Aware Label Assignment (YOLO26)
    stal_min_positives: int = 0  # minimum positive anchors per GT (0 = disabled)
    stal_backfill_mode: str = "distance"  # "distance" or "distance_cls"

    # Weighted sampling
    weighted_sampling: bool = False
    sampler_gamma: float = 0.85

    # Augmentation (polygon-safe)
    stain_jitter: bool = True  # HSV color jitter for histopathology
    stain_hsv_h: float = 0.05  # hue shift range
    stain_hsv_s: float = 0.3  # saturation scale range (±)
    stain_hsv_v: float = 0.2  # value/brightness scale range (±)
    stain_blur_prob: float = 0.2  # probability of Gaussian blur
    stain_blur_sigma: float = 1.0  # max blur sigma

    scale_augment: bool = True  # random scale augmentation
    scale_range: tuple[float, float] = (0.7, 1.3)  # min/max scale factors

    translate_augment: bool = True  # random translation augmentation
    translate_range: float = 0.1  # fraction of crop_size

    inference_conf: float = 0.20  # confidence threshold at inference

    refinement_kernel_size: int = 3

    # Early stopping
    patience: int = 100  # epochs with no improvement before stopping (default: 100)

    # GradNorm — dynamic per-task loss weighting
    gradnorm: bool = False  # enable GradNorm
    gradnorm_alpha: float = 0.5  # restoring force: 0=uniform, 1=aggressive
    gradnorm_warmup_epochs: int = 5  # use static weights for first N epochs  # kernel size for polygon refinement block

    # E2E dual-TAL assignment (NMS-free)
    tal_topk: int = 13  # one2many positives per GT
    assigner_alpha: float = 0.5  # cls^alpha in alignment metric
    assigner_beta: float = 6.0  # iou^beta in alignment metric
    nwd_enabled: bool = False   # replace pIoU with NWD (Wasserstein) similarity
    nwd_c: float = 0.001  # NWD normalisation constant (smaller = sharper)

    # Auxiliary xy head — bypass backbone→head bottleneck
    aux_xy_weight: float = 0.0  # Huber loss weight (0 = disabled, 10.0 = recommended)
    aux_xy_ramp_epochs: int = 100  # epochs over which aux weight decays

    # Prediction-level self-attention on top-K scored predictions (o2o branch only).
    # After FCN scores all 5376 anchors, top-K=100 by confidence are selected.
    # These K predictions (mostly fg) undergo TransformerEncoder self-attention,
    # producing per-prediction suppression weights. Fundamentally different from
    # feature-level attention (train31/32/34 dead ends) which operated on 5376
    # bg-dominated anchor features.
    prediction_refinement_weight: float = 0.0  # BCE loss weight (0 = disabled, 1.0 = recommended)
    prediction_refinement_topk: int = 100  # number of top predictions to refine

    # Range-based L1 loss (LSP-DETR-inspired): per-ray tolerance band for overlaps.
    # loss = max(r_gt*(1-eps) - r_pred, 0) + max(r_pred - r_gt*(1+eps), 0)
    # Zero if prediction falls within [r_gt*(1-eps), r_gt*(1+eps)].
    # r_max extends to infinity in overlap regions (natural overlap handling).
    # Supplements pIoU (kept for assignment metric). 0 = disabled.
    range_l1_weight: float = 0.0
    range_l1_eps: float = 0.1  # tolerance fraction (0.1 = ±10% of r_gt)

    # Asymmetric max() bound loss (LSP-DETR criterion.py:16-27):
    # loss = max(relu(r_gt*(1-eps) - r_pred), relu(r_pred - r_gt*(1+eps)))
    # Takes worst violation per ray instead of summing. 0 = disabled.
    bound_l1_weight: float = 0.0
    bound_l1_eps: float = 0.1  # tolerance fraction (0.1 = ±10% of r_gt)

    # Inter-scale competition: softmax across scales to suppress cross-scale duplicates.
    # Upsamples P3/P4 cls to P2 resolution, stacks, softmax across scale dim,
    # gathers regression from winning scale. Addresses multi-scale duplicates.
    inter_scale_competition: bool = False
    inter_scale_temperature: float = 1.0  # softmax temperature (lower = sharper competition)
    inter_scale_pixel_shuffle: bool = False  # sub-grid-aware competition via PixelShuffle (trainable)

    # Local (intra-scale) competition: per-scale 3×3 neighborhood softmax.
    # Each anchor competes with its 8 neighbors — local winner-take-more.
    # Suppresses same-scale duplicates where one nucleus fires adjacent anchors.
    local_competition: bool = False
    local_competition_kernel: int = 3  # neighborhood size (3 = 3×3 = 8 neighbors)
    local_competition_temperature: float = 1.0  # softmax temperature (lower = sharper)

    # Pretrained backbone
    pretrained_backbone: str | None = None  # path to pretrained .pt (e.g. 'yolo26s.pt')

    # DCNv2 in head
    dcn_in_reg_head: bool = False  # replace 2nd Conv in cv2 with modulated deformable conv
    dcn_in_cls_head: bool = False  # replace 2nd Conv in cv3 with modulated deformable conv

    # Hierarchical cls: split o2o cls into binary (fg/bg, 1ch) + class (cell type, nc ch).
    # Binary head gets ALL 5376 anchors of gradient → strong fg/bg discrimination.
    # Class head only learns inter-class separation on fg anchors.
    # At inference: sigmoid(binary) × softmax(class).
    hierarchical_cls: bool = False
    cls_only_tal: bool = False  # o2o: assign by cls*gauss (no pIoU)
    cls_only_anneal_epoch: int = 100  # linear pIoU→cls blend over N epochs
    hierarchical_cls_detach: bool = True  # Stop-grad binary head input to prevent backbone flooding
    hierarchical_binary_threshold: float = 0.01  # Binary gate threshold at inference
    hierarchical_soft_cascade: bool = True  # Weight class loss by binary head confidence (soft curriculum)
    nc_override: int | None = None  # Force nc (e.g. 1 for binary detection); remaps all labels to class 0

    # Gradient clipping (LSP-DETR uses 0.1). Ultralytics defaults to 10.0.
    clip_grad: float = 10.0

    # Feature Bank: EMA prototype bank for classification feature refinement
    # (Chen et al., 2026 — JCST). Maintains per-class running-mean prototypes
    # from the cls head's intermediate c3 features. A contrastive loss pulls
    # fg features toward their assigned class prototype.
    feature_bank_enabled: bool = False  # enable feature bank (0 = disabled)
    feature_bank_momentum: float = 0.9  # EMA momentum for prototype updates
    feature_bank_temperature: float = 0.07  # InfoNCE temperature
    feature_bank_weight: float = 0.5  # contrastive loss weight multiplier
    feature_bank_warmup_epochs: int = 0  # epochs to wait before activating FB (0 = immediate)

    model_config = ConfigDict(extra='forbid')


# === GLOBAL SETTINGS ===
class GlobalSettings(BaseModel):
    root_dir: str
    max_size: int = 1024
    crop_size: int = 640
    output_mpp: float
    patching_overlap_pct: float
    annotation_type: str = 'bbox'
    n_rays: int = 32
    model: str | None = None  # model YAML override (e.g. yolo26s-p2.yaml). If set, overrides variant.

    global_cell_map: dict[str, int] = Field(default_factory=dict)
    global_tissue_map: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_sizes(self):
        if self.crop_size < 1:
            raise ValueError(f'crop_size must be >= 1, got {self.crop_size}')
        if self.max_size < 1:
            raise ValueError(f'max_size must be >= 1, got {self.max_size}')
        return self


# === DATASET CONFIG ===
_VALID_INGESTORS = frozenset({'parquet', 'geojson', 'csv_poly', 'mat_inst'})


class DatasetConfig(BaseModel):
    root_dir: str
    native_mpp: float
    split_separation: Literal['physical', 'filename_regex', 'none']
    modality_separation: Literal['physical_parallel', 'physical_flat', 'bundled_archive']

    ingestor: str

    # Conditional fields
    split_dirs: dict[str, str] | None = None
    split_args: SplitArgs | None = None
    modality_dirs: ModalityDirs | None = None
    modality_pairing_rule: ModalityPairingRule | None = None
    csv_column_map: CSVColumnMap | None = None

    # Mappings
    namespace_map: dict[str, str] = Field(default_factory=dict)
    tissue_map: dict[str, str] = Field(default_factory=dict)
    tissue_type: str | None = None

    @model_validator(mode='after')
    def validate_ingestor(self):
        if self.ingestor not in _VALID_INGESTORS:
            raise ValueError(f'Unknown ingestor={self.ingestor!r}. Supported: {sorted(_VALID_INGESTORS)}')
        return self

    @model_validator(mode='after')
    def validate_split_separation_requirements(self):
        if self.split_separation == 'physical':
            if not self.split_dirs:
                raise ValueError("split_dirs required for split_separation='physical'")
            for key in self.split_dirs.keys():
                if not key.endswith('_dir'):
                    raise ValueError(f"split_dirs key '{key}' must end with '_dir'")

        elif self.split_separation == 'filename_regex':
            if not self.split_args or not self.split_args.regex:
                raise ValueError("split_args.regex required for split_separation='filename_regex'")
            if self.split_args.split_map:
                valid_splits = {'train', 'val', 'test'}
                for mapped in self.split_args.split_map.values():
                    if mapped not in valid_splits:
                        raise ValueError(
                            f"split_map value '{mapped}' is not a recognized split name. Must be one of: {valid_splits}"
                        )

        elif self.split_separation == 'none':
            if self.split_args:
                ratio_sum = self.split_args.train_ratio + self.split_args.val_ratio + self.split_args.test_ratio
                if abs(ratio_sum - 1.0) > 0.01:
                    raise ValueError(
                        f'split ratios must sum to ~1.0, got {ratio_sum:.3f} '
                        f'(train={self.split_args.train_ratio}, val={self.split_args.val_ratio}, '
                        f'test={self.split_args.test_ratio})'
                    )
        return self

    @model_validator(mode='after')
    def validate_modality_separation_requirements(self):
        if self.modality_separation == 'physical_parallel':
            if not self.modality_dirs:
                raise ValueError("modality_dirs required for modality_separation='physical_parallel'")

        elif self.modality_separation == 'physical_flat':
            if not self.modality_pairing_rule or not self.modality_pairing_rule.match_extension:
                raise ValueError(
                    "modality_pairing_rule.match_extension required for modality_separation='physical_flat'"
                )
        return self


# === FULL CONFIG ===
class ETLConfigModel(BaseModel):
    global_settings: GlobalSettings
    datasets: dict[str, DatasetConfig]
    training: TrainingSettings | None = None
    namespace_map: dict[str, dict[str, str]] = Field(default_factory=dict)


# === UPDATED ETLConfig CLASS ===
class ETLConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        raw_config = self._load_yaml()

        # Parse & validate with Pydantic
        self.model = ETLConfigModel(**raw_config)

        # Expose attributes for backward compatibility
        self.global_settings = self.model.global_settings.model_dump()
        self.datasets = {k: v.model_dump() for k, v in self.model.datasets.items()}
        self.namespace_map = self.model.namespace_map
        self.training = self.model.training.model_dump() if self.model.training else None

    def _load_yaml(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f'Configuration file not found: {self.config_path}')
        with open(self.config_path) as file:
            return yaml.safe_load(file)

    def get_dataset_config(self, dataset_name: str) -> dict[str, Any]:
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration.")

        d_conf = self.datasets[dataset_name].copy()

        # Safely resolve the root directory
        global_root = Path(self.global_settings.get('root_dir', '.')).resolve()
        dataset_root = Path(d_conf.get('root_dir', ''))
        resolved_root = global_root.joinpath(dataset_root)
        d_conf['root_dir'] = str(resolved_root)

        # Merge configs: Global is the base, Dataset overrides the base
        merged_conf = self.global_settings.copy()
        merged_conf.update(d_conf)

        return merged_conf

    def get_global_config(self) -> dict[str, Any]:
        return self.global_settings

    def list_datasets(self) -> list[str]:
        return list(self.datasets.keys())

    def get_namespace_map(self, dataset_name: str = None) -> dict:
        if dataset_name:
            return self.namespace_map.get(dataset_name, {})
        return self.namespace_map
