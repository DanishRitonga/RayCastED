from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl


class BaseDataIngestor(ABC):
    """Abstract base class for data ingestors.

    Handles file discovery, pairing, and split assignment based on a flexible
    configuration.
    """

    def __init__(self, config: dict[str, Any]):
        """Initializes the ingestor, parses the config, and builds the file registry."""
        self.root_dir = Path(config.get('root_dir'))
        self.config = config
        self.file_registry: pl.DataFrame = None

        # Dataset-level string-to-string maps
        self.namespace_map = config.get('namespace_map', {})
        self.tissue_map = config.get('tissue_map', {})

        # Global string-to-integer maps
        self.global_cell_map = config.get('global_cell_map', {})
        self.global_tissue_map = config.get('global_tissue_map', {})

        # Spatial Harmonization Setup
        self.target_mpp = config.get('output_mpp')
        self.native_mpp = config.get('native_mpp')

        # The dynamic annotation routing key (defaults to 'bbox')
        self.annotation_type = config.get('annotation_type', 'bbox').lower()

        if self.target_mpp and self.native_mpp:
            self.scale_factor = self.native_mpp / self.target_mpp
        else:
            self.scale_factor = 1.0

        # Build the registry immediately upon instantiation
        self._build_registry()

    def _build_registry(self):
        """The orchestrator method that discovers files, pairs them, and assigns splits."""
        records = []

        # ---------------------------------------------------------
        # STEP 1 & 2: Discover and Pair Files based on split/modality
        # ---------------------------------------------------------
        split_sep = self.config.get('split_separation')
        mod_sep = self.config.get('modality_separation')

        if split_sep == 'physical':
            # Search inside specific Train/Val/Test/Fold directories
            split_dirs = self.config.get('split_dirs', {})
            for split_key, split_rel_path in split_dirs.items():
                if not split_key.endswith('_dir'):
                    raise ValueError(f"Invalid split key: {split_key}. Must end with '_dir'")

                clean_split_label = split_key.replace('_dir', '')
                search_base = self.root_dir / split_rel_path

                records.extend(self._scan_and_pair(search_base, mod_sep, split_label=clean_split_label))

        else:
            # Search the whole root directory (Regex or None)
            records.extend(self._scan_and_pair(self.root_dir, mod_sep, split_label=None))

        # Convert to Polars DataFrame for fast, vectorized operations later
        df = pl.DataFrame(
            records, schema={'roi_id': pl.Utf8, 'image_path': pl.Utf8, 'mask_path': pl.Utf8, 'split': pl.Utf8}
        )

        # ---------------------------------------------------------
        # STEP 3: Apply Regex Split Tagging (if applicable)
        # ---------------------------------------------------------
        if split_sep == 'filename_regex':
            regex_pattern = self.config.get('split_args', {}).get('regex', '')
            if not regex_pattern:
                raise ValueError('split_args.regex must be provided for filename_regex split_separation.')

            # Polars native regex extraction
            df = df.with_columns(pl.col('image_path').str.extract(regex_pattern, 1).alias('split'))
        elif split_sep == 'none':
            df = df.with_columns(pl.lit('unassigned').alias('split'))

        # Drop any rows where split couldn't be determined or file pairing failed
        self.file_registry = df.drop_nulls()

    def _scan_and_pair(self, base_path: Path, mod_sep: str, split_label: str = None) -> list:
        """Handles the actual file discovery and RGB-to-Mask pairing logic."""
        records = []

        if mod_sep == 'bundled_archive':
            # Method 1 (Parquet) - Image and Mask are the exact same file
            for file_path in base_path.rglob('*.parquet'):
                records.append(
                    {
                        'roi_id': file_path.stem,
                        'image_path': str(file_path),
                        'mask_path': str(file_path),  # Points to same archive
                        'split': split_label,
                    }
                )

        elif mod_sep == 'physical_parallel':
            # Method 2, 3, 4, 5 - Images and Masks are in separate places
            mod_dirs = self.config.get('modality_dirs', {})
            img_dir = base_path / mod_dirs.get('image_dir', '')
            mask_dir = base_path / mod_dirs.get('mask_dir', '')

            pairing_rule = self.config.get('modality_pairing_rule', {})
            target_ext = pairing_rule.get('match_extension', '')
            suffix_to_replace = pairing_rule.get('suffix_to_replace', '')
            add_suffix = pairing_rule.get('add_suffix', '')

            # Assume images are PNG/TIF unless specified
            for img_path in img_dir.rglob('*.*'):
                if img_path.suffix not in ['.png', '.tif', '.tiff', '.jpg']:
                    continue

                # Construct expected mask name
                roi_id = img_path.stem
                mask_stem = roi_id

                if suffix_to_replace:
                    mask_stem = mask_stem.replace(suffix_to_replace, '')
                if add_suffix:
                    mask_stem += add_suffix

                expected_mask_name = f'{mask_stem}{target_ext}'
                expected_mask_path = mask_dir / expected_mask_name

                if expected_mask_path.exists():
                    records.append(
                        {
                            'roi_id': roi_id,
                            'image_path': str(img_path),
                            'mask_path': str(expected_mask_path),
                            'split': split_label,
                        }
                    )
                else:
                    print(f'Warning: Missing mask for {img_path.name}. Skipping.')

        return records

    def get_registry(self, split: str = None) -> pl.DataFrame:
        """Returns the registry, optionally filtered by split (train, val, test)."""
        if split:
            return self.file_registry.filter(pl.col('split') == split)
        return self.file_registry

    def standardize_label(self, raw_label: Any) -> int:
        """Two-step translation: Raw String -> Standard String -> Global Integer."""
        raw_str = str(raw_label)

        # Step 1: Translate raw dataset label to standardized string
        if raw_str not in self.namespace_map:
            raise ValueError(f"Dataset Mapping Error: Unknown raw label '{raw_str}'. Update dataset 'namespace_map'.")
        standard_str = self.namespace_map[raw_str]

        # Step 2: Translate standardized string to global integer
        if standard_str not in self.global_cell_map:
            raise ValueError(f"Global Mapping Error: Standard label '{standard_str}' not found in 'global_cell_map'.")

        return self.global_cell_map[standard_str]

    def resolve_tissue(self, raw_tissue_id: Any = None) -> int:
        """Two-step translation: Raw ID/Type -> Standard Tissue String -> Global Integer."""
        standard_tissue_str = 'unknown_tissue'

        # Step 1: Resolve the standard string
        if 'tissue_type' in self.config and self.config['tissue_type'] is not None:
            standard_tissue_str = str(self.config['tissue_type'])
        elif raw_tissue_id is not None:
            raw_str = str(raw_tissue_id)
            if raw_str not in self.tissue_map:
                raise ValueError(f"Tissue Mapping Error: Unknown tissue ID '{raw_str}'. Update dataset 'tissue_map'.")
            standard_tissue_str = self.tissue_map[raw_str]

        # Step 2: Translate to global integer
        if standard_tissue_str not in self.global_tissue_map:
            raise ValueError(f"Global Mapping Error: Tissue '{standard_tissue_str}' not found in 'global_tissue_map'.")

        return self.global_tissue_map[standard_tissue_str]

    def standardize_mpp(self, image: np.ndarray, annotations: Any) -> tuple[np.ndarray, Any]:
        """Scales the RGB image and its corresponding annotations based on the globally defined annotation_type."""
        if self.scale_factor == 1.0:
            return image, annotations

        h, w = image.shape[:2]
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)

        # 1. Scale the Image (Smooth interpolation for biology)
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 2. Route the Annotation Scaling Math
        scaled_annotations = None

        if len(annotations) == 0 and self.annotation_type != 'instance_mask':
            # Handle empty arrays for coordinate-based types
            scaled_annotations = annotations

        elif self.annotation_type == 'bbox':
            # Format: (N, 5) array -> [xmin, ymin, xmax, ymax, class_id]
            scaled_annotations = annotations.copy()
            scaled_annotations[:, :4] = np.round(scaled_annotations[:, :4] * self.scale_factor).astype(np.int32)
            scaled_annotations[:, [0, 2]] = np.clip(scaled_annotations[:, [0, 2]], 0, new_w)
            scaled_annotations[:, [1, 3]] = np.clip(scaled_annotations[:, [1, 3]], 0, new_h)

        elif self.annotation_type == 'instance_mask':
            # Format: (H, W) integer matrix
            # STRICT Requirement: INTER_NEAREST prevents ID corruption
            scaled_annotations = cv2.resize(annotations, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        elif self.annotation_type == 'polygon':
            # Format: List of arrays/lists, where each is [x1, y1, x2, y2, ..., class_id]
            scaled_annotations = []
            for poly in annotations:
                poly_arr = np.array(poly, dtype=np.float32)
                # Scale all elements except the final class_id
                poly_arr[:-1] = np.round(poly_arr[:-1] * self.scale_factor)
                scaled_annotations.append(poly_arr.astype(np.int32))

        elif self.annotation_type == 'raycast':
            # Format: (N, 35) -> [class_id, cx, cy, d_1, ..., d_32]
            scaled_annotations = annotations.copy()
            # Scale cx, cy and ray distances. class_id at index 0 is untouched.
            scaled_annotations[:, 1:] = scaled_annotations[:, 1:] * self.scale_factor

        else:
            raise ValueError(f"Unsupported annotation_type: '{self.annotation_type}'")

        return scaled_image, scaled_annotations

    @abstractmethod
    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, int]:
        """Abstract method to process found data."""
        pass
