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

    # NuLite architecture (FastViT encoder + NuLite decoder + NP seg head + PAN + SAM conditioning)
    architecture: str = 'yolo26'  # 'yolo26' = classic YAML-built FCN; 'nulite' = NuLiteRayCastModel
    nulite_variant: str = 'fastvit_s12'  # FastViT encoder variant (timm create_model name)
    nulite_ckpt: str | None = None  # NuLite-pretrained encoder/decoder ckpt (V2 upper bound); null = from-scratch
    nulite_oracle_kind: str = 'binary'  # binary, center_potential, or fg_centroid
    pretrained: bool = True  # ImageNet-pretrained encoder via timm (V1/V2); false = full scratch
    lambda_seg: float = 1.0  # NP seg loss weight (0.0 = V3 no-seg baseline)
    seed_map_target: bool = False  # InstanSeg-style center-weighted distance-to-boundary seg target (vs hard binary NP)
    gate_scale: float = 1.0  # SAM gate bound (cond = det + gate_scale*fuse); <1.0 forces head to learn discrimination
    freeze_encoder_decoder: bool = False  # phase-2a: freeze encoder+decoder+NP, train head only

    # NuLite DETR architecture (single query-based head, seed-map query selection)
    detr_nq: int = 300  # number of queries (dense tiles reach ~200 nuclei)
    detr_ndl: int = 3  # decoder layers (lightweight DETR recipe: shallow decoder)
    detr_hd: int = 256  # decoder hidden dim
    detr_d_ffn: int = 1024  # decoder FFN hidden dim (head-slimming: 512 = ~16% head params off)
    detr_seed_threshold: float = 0.5  # seed-map local-maxima min value for query seeding
    detr_peak_distance: int = 3  # seed-map local-maxima neighbourhood (px)
    detr_grid_size: float = 0.05  # grid cell diameter for grid-query init (normalized); ~3.5um at 256px
    detr_query_selection: str = 'grid'  # 'grid' (LSP-DETR fixed grid), 'learned' (enc_score_head top-k), or 'seed' (local maxima)
    detr_query_warmup_epochs: int = 0  # >0 with query_selection=seed: run 'grid' for N epochs then switch to seed
    detr_seed_feature: bool = True  # concat seed map as extra feature channel (seg conditioning)
    detr_no_object: bool = True  # explicit "no object" class via focal loss (LSP-DETR style)
    detr_seed_in_content: bool = True  # concat raw seed logit to score-head inputs (undiluted objectness read)
    detr_no_object_weight: float = 1.0  # ∅ focal column weight (>1 = suppress redundant queries harder)
    detr_cls_weight: float = 2.0  # cls loss gain for the DETR criterion (loss_gain['class'])
    detr_piou_weight: float = 3.0  # polar-IoU loss gain for the DETR criterion (loss_gain['piou'])
    detr_xy_weight: float = 500.0  # centroid loss gain (loss_gain['xy']); 500 was tuned for Huber delta=0.05
    detr_lsp: bool = False  # LSP-DETR-exact loss/matching+query init (equal ∅ cols, no pIoU, 9999 gate)
    detr_matcher_lsp: bool = False  # LSP-DETR-exact MATCHER only (class1/pointL1/radial-from-centroid/9999 gate); loss + query init stay locked
    detr_query_block_size: float = 0.0  # grid cell dia in PIXELS for LSP init (0 = legacy grid_size grid)
    detr_content_top: bool = False  # query content from top-level (st32) feature (LSP FeatureSampling)
    detr_seed_resize: str = 'bilinear'  # F-channel seed downsample: bilinear (locked) | max | avg | topk
    detr_seed_resize_frac: float = 0.25  # 'topk' only: fraction of cell pixels averaged (1/k_cell at each scale)
    detr_layer_schedule: str = ''  # per-layer cross-attn scale: 0=b3,1=b4,2=b5; '2,1,0,2,1,0'; '' = all levels
    detr_hybrid_peaks: int = 100  # query_selection=hybrid: extra seed-peak queries appended to the grid
    detr_seed_level: bool = False  # seed map as 4th cross-attn level (own strided proj) instead of +1 F channel
    detr_seed_level_stride: int = 4  # seed-level proj conv stride (4 = b3 res; smaller = finer/expensive)
    detr_recast: bool = False  # local embedding correction between existing decoder layers
    detr_recast_after_layer: int = 3  # one-based; must precede final layer
    detr_recast_radius_px: float = 4.0  # nine local samples spanning +/- radius in input pixels
    detr_respawn: bool = False  # recall-repair 2nd decoder pass: re-anchor emptiest queries on uncovered seed peaks
    detr_respawn_peaks: int = 64  # max respawned queries per image
    detr_respawn_min_seed: float = 0.5  # min seed value for a candidate respawn peak
    detr_self_attention_rope: bool = False  # rotate decoder self-attention Q/K at current query centroids
    detr_rope_theta: float = 100.0  # STRING frequency base used by LSP-DETR
    detr_rope_position_scale: float = 0.05  # normalized spatial unit (one 20x20 grid cell)
    detr_polygon_masked_attention: bool = False  # prior polygon softly masks next layer's cross-attention samples
    detr_polygon_oracle_mask: bool = False  # multiply the query polygon mask by frozen oracle FG probability
    detr_polygon_mask_floor: float = 0.1  # minimum outside-polygon attention multiplier
    detr_polygon_mask_temperature: float = 0.1  # outside-boundary decay width in radius-ratio units
    detr_appearance_weight: float = 0.0  # stride-2 union-mask BCE+Dice and query appearance injection
    detr_appearance_channels: int = 32  # supervised high-resolution appearance width
    detr_presence_weight: float = 0.0  # training-only post-fusion query presence BCE
    detr_appearance_imitation_weight: float = 0.0  # owner-only small-query exemplar loss
    detr_appearance_imitation_temperature: float = 0.1
    detr_appearance_imitation_small_area: float = 200.0
    detr_presence_negative_margin_px: float = 4.0  # unambiguous negative center clearance
    detr_aux_dense_weight: float = 0.0  # training-only dense supervision on trainable b3/b4/b5 (0=off)
    detr_aux_dense_type_weight: float = 1.0  # nucleus-type term relative to dense objectness
    detr_seg_fg_margin_weight: float = 0.0  # matched-positive segmentation-guided true-class vs empty margin
    detr_seg_fg_margin: float = 1.0  # required z_true - z_empty margin when segmentation supports foreground
    detr_seg_fg_radius_px: float = 4.0  # local 3x3 segmentation evidence radius around matched query centroid
    detr_seg_fg_margin_layer_weights: list[float] = Field(
        default_factory=list
    )  # per-decoder-layer multipliers; [] = final layer only
    backbone_freeze_epochs: int = 0  # freeze FastViT encoder for N initial epochs (LSP-DETR: 30); 0 = end-to-end
    backbone_lr_scale: float = 0.1  # encoder LR multiplier after unfreeze (LSP-DETR: backbone at 10% base LR)
    decoder_freeze_epochs: int = 0  # freeze the NuLite upsampling decoder (encoder_decoder.* minus encoder) for N initial epochs (curriculum stage-2 warm-up); 0 = train decoder from ep0
    decoder_lr_scale: float = 0.1  # decoder LR multiplier after its freeze window ends (guards oracle-initialised decoder from head gradients)
    detr_mds: bool = True  # MDS-DETR rank-causal self-attn mask (local winner-take-all, duplicate suppression)
    detr_cost_inside: float = 10.0  # LSP-DETR inner-mask cost Wm: lambda for matching a centroid outside its GT (0=off)
    detr_per_layer_match: bool = False  # LSP-DETR 'look forward twice': run the Hungarian matcher per decoder layer (vs reuse last-layer match)
    detr_duplicate_train_topk: int = 1  # positives per GT; 1=Hungarian owner only, 3=owner plus <=2 duplicates
    detr_duplicate_nms_iou: float = 0.5  # polygon NMS IoU used only when duplicate_train_topk > 1
    detr_use_decoder: bool = True  # False = bypass NuLite decoder+NP head, DETR reads raw FastViT stage features (~12.2M)
    detr_shared_layers: bool = False  # weight-share the transformer decoder layer across all ndl steps (params-only saving)
    detr_separable_decoder: bool = False  # depthwise-separable NuLite decoder convs (sub-10M model)
    detr_backbone: str = 'nulite'  # 'nulite' (FastViT + NuLite decoder) or 'yolo11' (YOLO11n + FPN top-down)
    det_variant: str = 'fastvit_t12'  # detection-stream encoder variant for detr_backbone=two_stream (seg stream uses nulite_variant)
    detr_use_seed: bool = True  # add stride-4 NP seed head on P2 (yolo11 backbone only)
    yolo11_scale: str = 'n'  # yolo11 variant scale (n/s/m/l/x) for detr_backbone=yolo11
    yolo11_ckpt: str | None = None  # path to yolo11*.pt for layers 0-16 (None = auto 'yolo11n.pt' when pretrained)
    fitness_warmup_epochs: int = 0  # mask fitness before this epoch (guards vs early fluke spikes)

    # Mixed precision. DETR (transformer) training is fp16-overflow prone
    # (observed: GradScaler scale → 0.0 → all-NaN at epoch ~87); set false to
    # train the DETR architecture in fp32.
    amp: bool = True

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

    # Pixel-Level Balancing — area-based fg weighting to boost small nuclei
    plb_enabled: bool = False

    # Per-o2o classification loss overrides — the o2o branch must be a
    # "background specialist" because NMS-free inference has no safety net.
    # Set to None to inherit the base value.
    focal_gamma_o2o: float | None = None
    focal_alpha_o2o: float | None = None
    bg_fg_ratio_o2o: int | None = None

    # Loss weights (literature: regression should be 3-7x higher than cls)
    lambda_l1: float = 14.0  # L1 ray distance weight
    lambda_piou: float = 13.0  # Polar IoU weight
    lambda_cls: float = 2.0  # classification weight
    lambda_xy: float = 500.0  # centroid xy weight

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

    # Extra photometric aug (annotation-safe, LSP-DETR parity)
    aug_noise_prob: float = 0.0  # probability of additive Gaussian noise (0 = off)
    aug_noise_std: float = 0.1  # noise std in [0, 1] image units

    # Random-sized crop: random sub-window resized back to crop_size (zoom-in)
    aug_randcrop_prob: float = 0.0  # probability (0 = off)
    aug_randcrop_min: int = 128  # min crop side (px)
    aug_randcrop_max: int = 256  # max crop side (px)

    inference_conf: float = 0.20  # confidence threshold at inference

    # Early stopping
    patience: int = 100  # epochs with no improvement before stopping (default: 100)

    # E2E dual-TAL assignment (NMS-free)
    tal_topk: int = 13  # one2many positives per GT
    assigner_alpha: float = 0.5  # cls^alpha in alignment metric
    assigner_beta: float = 6.0  # iou^beta in alignment metric

    # Auxiliary xy head — bypass backbone→head bottleneck
    aux_xy_weight: float = 0.0  # Huber loss weight (0 = disabled, 10.0 = recommended)
    aux_xy_ramp_epochs: int = 100  # epochs over which aux weight decays

    # Knowledge distillation: o2m cls heads → o2o cls heads.
    # BCE loss between detached o2m sigmoid probs and o2o logits on ALL anchors.
    # Fixes gradient starvation: o2o cls sees ~28 fg with bg_fg_ratio_o2o=0
    # vs o2m's 420 fg (15x). Self-distillation gives o2o the full 5376-anchor
    # soft signal from a richer head without changing 1:1 assignment.
    o2o_distill_weight: float = 0.0  # 0=disabled, 0.5=recommended

    # Inter-scale competition: softmax across scales to suppress cross-scale duplicates.
    # Upsamples P3/P4 cls to P2 resolution, stacks, softmax across scale dim,
    # gathers regression from winning scale. Addresses multi-scale duplicates.
    inter_scale_competition: bool = False
    inter_scale_temperature: float = 1.0  # softmax temperature (lower = sharper competition)

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
    hierarchical_cls_detach: bool = True  # Stop-grad binary head input to prevent backbone flooding
    hierarchical_binary_threshold: float = 0.01  # Binary gate threshold at inference
    nc_override: int | None = None  # Force nc (e.g. 1 for binary detection); remaps all labels to class 0

    # Analytical rays: compute GT ray targets from predicted centroid → polygon
    # boundary via Cramer's rule instead of using precomputed static GT rays.
    # Couples centroid and ray gradients: ∂L/∂centroid ≠ 0 through chain rule.
    # Gradient clipping (LSP-DETR uses 0.1). Ultralytics defaults to 10.0.
    clip_grad: float = 10.0

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
