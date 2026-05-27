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

    # Learning rate
    cos_lr: bool = True  # cosine LR schedule. Was False

    # Assigner
    assigner_radius_scale: float = 2.0  # containment radius multiplier. Was 1.5

    # Classification loss — Focal or BCE
    # Previous "focal tested, harmful" result was caused by alpha=1.0 which
    # zeroed all background gradients. Standard alpha=0.25 works correctly.
    focal_gamma: float = 0.0  # 0 = BCE, >0 = Focal loss (recommended: 2.0)
    focal_alpha: float = 0.25  # positive weight (standard: 0.25)
    log_ray_loss: bool = False  # log-space L1 on rays for scale-invariant errors
    bg_fg_ratio: int = 3  # max bg anchors per fg anchor in cls loss

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
    bg_fg_ratio_o2o_curriculum_epoch: int = 0  # epoch to start ramping bg_fg_ratio_o2o from 0→bg_fg_ratio_o2o
    bg_cls_decay_o2o: float | None = None  # per-branch bg suppression for o2o
    fg_cls_boost_o2o: float | None = None  # per-branch quality re-weighting for o2o
    fg_cls_quality_scale_o2o: float | None = None  # per-branch multiplicative quality for o2o

    # Loss weights (literature: regression should be 3-7x higher than cls)
    lambda_l1: float = 14.0  # L1 ray distance weight
    lambda_piou: float = 13.0  # Polar IoU weight
    lambda_cls: float = 2.0  # classification weight
    lambda_xy: float = 500.0  # centroid xy weight

    # Unified suppression loss (quality ranking + spatial repulsion)
    lambda_suppress: float = 0.0  # 0 = disabled
    suppress_radius: float = 0.05  # normalised repulsion radius (~13px at 256px)

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

    # 2-phase Hungarian curriculum: TAL-only → Hungarian ramp
    # Phase 1 (0 → phase2_start): standard dual-TAL, o2m > o2o
    # Phase 2 (phase2_start → end): Hungarian o2o ramps 0→hungarian_max_weight
    hungarian_phase2_start: int = 100
    hungarian_max_weight: float = 0.9
    hungarian_ramp_epochs: int = 0
    phase2_freeze_epochs: int = 0
    hungarian_cost_class: float = 1.0
    hungarian_cost_centroid: float = 1.0
    hungarian_cost_ray: float = 1.0
    hungarian_cost_inner: float = 9999.0  # Outside-polygon penalty (AND gate: 0=disabled, 9999=LSP-DETR default)
    hungarian_cost_ray_quality: float = 1.0  # Extra ray weight for quality emphasis (soft, not hard piou gate)
    hungarian_cost_inner_sigma: float = 0.1  # Soft boundary width for cost_inner (0→hard gate, 0.1→smooth)
    hungarian_cls_only: bool = True

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

    # DINO-style contrastive denoising (o2o branch only)
    # Injects corrupted GT copies before TAL assignment, providing more
    # fg training signal for the o2o cls head. With QFL soft targets,
    # corrupted copies get lower quality scores → contrastive learning.
    dn_num: int = 0  # number of corrupted copies per GT (0 = disabled)
    dn_centroid_noise: float = 0.0  # max centroid shift in normalised coords
    dn_ray_noise: float = 0.0  # multiplicative Gaussian ray jitter std

    # Auxiliary xy head — bypass backbone→head bottleneck
    aux_xy_weight: float = 0.0  # Huber loss weight (0 = disabled, 10.0 = recommended)
    aux_xy_ramp_epochs: int = 100  # epochs over which aux weight decays

    # Quality head — IoU-aware inference scoring
    quality_head_weight: float = 0.0  # L1 loss weight for piou prediction (0 = disabled, 1.0 = recommended)

    # Self-attention on o2o cls features — spatial context for duplicate suppression
    self_attention: bool = False  # per-scale linear self-attention before final cls projection
    cross_scale_attention: bool = False  # cross-scale attention on concatenated features across P2/P3/P4

    # PSS (Positional Suppression Structure) head — learned per-pixel suppression
    # 1-channel conv on o2o cls features: final_conf = cls × sigmoid(pss).
    # Trained with BCE: target=1 for best anchor per GT, 0 for all others.
    # Unlike quality head (absolute piou), PSS learns relative competition — which
    # position should "win" in each local neighborhood.
    pss_head_weight: float = 0.0  # BCE loss weight (0 = disabled, 1.0 = recommended)

    # Gaussian spatial soft targets — replace hard 0/1 cls targets with spatial Gaussian
    # Best anchor per GT → target 1.0; other fg anchors → exp(-d²/2σ²) where d is
    # distance to GT centroid in normalized space. Uses QFL for continuous targets.
    # Fixes train25/26/27 failure: the problem was piou-based targets (quality spread),
    # not the Gaussian concept. Spatial Gaussian creates natural "winner-take-most" gradient.
    gaussian_soft_targets: bool = False  # enable Gaussian spatial soft targets for o2o cls
    gaussian_sigma: float = 0.5  # spatial Gaussian sigma in normalized coords (0.5 ≈ 128px at 256)

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

    # Inter-scale competition: softmax across scales to suppress cross-scale duplicates.
    # Upsamples P3/P4 cls to P2 resolution, stacks, softmax across scale dim,
    # gathers regression from winning scale. Addresses multi-scale duplicates.
    inter_scale_competition: bool = False
    inter_scale_temperature: float = 1.0  # softmax temperature (lower = sharper competition)

    # Pretrained backbone
    pretrained_backbone: str | None = None  # path to pretrained .pt (e.g. 'yolo26s.pt')

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
