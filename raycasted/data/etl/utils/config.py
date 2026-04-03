from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# === NESTED MODELS FOR COMPLEX OBJECTS ===
class SplitArgs(BaseModel):
    regex: str | None = None


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


# === GLOBAL SETTINGS ===
class GlobalSettings(BaseModel):
    root_dir: str
    output_image_size: list[int]
    output_mpp: float
    patching_overlap_pct: float
    annotation_type: str = 'bbox'

    global_cell_map: dict[str, int] = Field(default_factory=dict)
    global_tissue_map: dict[str, int] = Field(default_factory=dict)


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
