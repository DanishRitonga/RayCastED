from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


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

    # Learning rate
    cos_lr: bool = True  # cosine LR schedule. Was False

    # Assigner
    assigner_topk: int = 20  # positive anchors per GT. Was 13
    assigner_radius_scale: float = 2.0  # containment radius multiplier. Was 1.5

    # Classification loss
    focal_loss: bool = True  # use focal loss instead of BCE. Was False
    focal_gamma: float = 2.0  # focal loss gamma
    quality_focal_loss: bool = False  # QFL: IoU-aware classification. Replaces BCE/focal when true.

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

    # Soft polar centerness (PolarMask++)
    soft_polar_centerness: bool = True  # add centerness prediction branch
    centerness_weight: float = 1.0  # BCE loss weight for centerness term


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
class DatasetConfig(BaseModel):
    root_dir: str
    ingestion_method: int
    native_mpp: float
    split_separation: Literal['physical', 'filename_regex', 'none']
    modality_separation: Literal['physical_parallel', 'physical_flat', 'bundled_archive']

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
