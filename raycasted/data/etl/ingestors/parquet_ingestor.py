from collections.abc import Generator

import cv2
import numpy as np
import polars as pl
from shapely.geometry import Polygon

from ..ops import polygon_to_raycast
from ._base import BaseDataIngestor


class ParquetIngestor(BaseDataIngestor):  # noqa: D101
    def process_item(self, row: dict) -> Generator:
        """Takes a registry row representing a Parquet file, extracts the ROIs,
        and yields them in the standardized format based on annotation_type.
        """  # noqa: D205
        parquet_path = row['image_path']
        base_roi_name = row['roi_id']

        lf = pl.scan_parquet(parquet_path)
        schema = lf.collect_schema()

        rgb_col, mask_col, cat_col, tissue_col = self._identify_columns(schema)

        if not all([rgb_col, mask_col, cat_col, tissue_col]):
            raise ValueError(
                f"""Could not map all columns in {parquet_path}. 
                RGB: {rgb_col}, Masks: {mask_col}, Cats: {cat_col}, Tissue: {tissue_col}"""
            )

        lf = lf.with_row_index('internal_roi_id')

        # Table 1: RGB Images
        df_rgb = lf.select(['internal_roi_id', rgb_col, tissue_col]).collect()

        # Table 2: Masks & Categories (Exploded)
        df_masks = lf.select(['internal_roi_id', mask_col, cat_col]).explode([mask_col, cat_col]).drop_nulls().collect()

        # Cleaner, faster Polars dictionary conversion
        masks_by_roi = {}
        if not df_masks.is_empty():
            # as_dict=True returns a dict where keys are tuples (e.g., (0,), (1,))
            raw_dict = df_masks.partition_by('internal_roi_id', as_dict=True)
            masks_by_roi = {k[0]: v for k, v in raw_dict.items()}

        # Now iterate through the RGB images and route based on annotation_type
        for rgb_row in df_rgb.iter_rows(named=True):
            internal_id = rgb_row['internal_roi_id']

            rgb_struct = rgb_row[rgb_col]
            rgb_bytes = rgb_struct['bytes'] if isinstance(rgb_struct, dict) else rgb_struct

            raw_tissue_id = rgb_row[tissue_col]
            tissue_origin = self.resolve_tissue(raw_tissue_id)

            image_array = self._decode_image(rgb_bytes, is_mask=False)

            global_roi_id = f'{base_roi_name}_roi_{internal_id}'

            # Route to appropriate annotation extractor based on annotation_type
            if self.annotation_type == 'bbox':
                annotations_array, cat_array = (
                    self._extract_bbox_annotations(masks_by_roi.get(internal_id), mask_col, cat_col, image_array),
                    None,
                )
            elif self.annotation_type == 'instance_mask':
                annotations_array, cat_array = self._extract_ins_segmentation_annotations(
                    masks_by_roi.get(internal_id), mask_col, cat_col, image_array
                )
            elif self.annotation_type == 'raycast':
                annotations_array, cat_array = (
                    self._extract_raycast_annotations(masks_by_roi.get(internal_id), mask_col, cat_col, image_array),
                    None,
                )
            else:
                raise ValueError(f'Unsupported annotation_type: {self.annotation_type}')

            # Apply common post-processing
            image_array, annotations_array = self.standardize_mpp(image_array, annotations_array)

            # Yield based on annotation type
            if cat_array is not None:
                yield (global_roi_id, image_array, annotations_array, cat_array, tissue_origin)
            else:
                yield (global_roi_id, image_array, annotations_array, tissue_origin)

    def _identify_columns(self, schema: pl.Schema) -> tuple[str, str, str, str]:
        rgb_col, mask_col, cat_col, tissue_col = None, None, None, None

        for col_name, dtype in schema.items():
            if isinstance(dtype, pl.Struct) or dtype == pl.Binary:
                rgb_col = col_name
            elif isinstance(dtype, pl.List) and (isinstance(dtype.inner, pl.Struct) or dtype.inner == pl.Binary):
                mask_col = col_name
            elif isinstance(dtype, pl.List) and dtype.inner in [pl.Int64, pl.Int32, pl.UInt32, pl.Int8]:
                cat_col = col_name
            # The new check: A standalone integer column
            elif dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.Int8] and not isinstance(dtype, pl.List):
                tissue_col = col_name

        return rgb_col, mask_col, cat_col, tissue_col

    def _decode_image(self, byte_string: bytes, is_mask: bool = False) -> np.ndarray:
        """Helper to convert raw bytes back into numpy arrays using OpenCV."""
        # Convert bytes to a 1D uint8 numpy array
        np_arr = np.frombuffer(byte_string, np.uint8)

        # Decode the array. Use IMREAD_UNCHANGED for masks to preserve exact values (e.g., boolean/binary masks),
        # use IMREAD_COLOR for RGB images to ensure 3 channels.
        flags = cv2.IMREAD_UNCHANGED if is_mask else cv2.IMREAD_COLOR
        decoded_img = cv2.imdecode(np_arr, flags)

        if decoded_img is None:
            raise ValueError('OpenCV failed to decode the byte array.')

        # OpenCV loads images in BGR format by default. Convert to RGB.
        if not is_mask and len(decoded_img.shape) == 3:
            decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

        return decoded_img

    def _extract_bbox_annotations(
        self, roi_masks_df, mask_col: str, cat_col: str, image_array: np.ndarray
    ) -> np.ndarray:
        """Extracts bounding boxes from mask contours.

        Returns an array of shape (N, 5) where each row is [xmin, ymin, xmax, ymax, class_id]
        """
        bboxes = []

        if roi_masks_df is not None:
            for mask_row in roi_masks_df.iter_rows(named=True):
                mask_struct = mask_row[mask_col]
                mask_bytes = mask_struct['bytes'] if isinstance(mask_struct, dict) else mask_struct

                category = mask_row[cat_col]
                mask_array = self._decode_image(mask_bytes, is_mask=True)

                if mask_array.ndim > 2:
                    mask_array = mask_array[:, :, 0]

                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Grab the bounding box of the largest contour
                    x, y, w, h = cv2.boundingRect(contours[0])
                    class_id = self.standardize_label(category)

                    bboxes.append([class_id, x, y, x + w, y + h])

        # Safe bounding box array initialization to guarantee (N, 5) shape
        bboxes_array = np.array(bboxes, dtype=np.int32) if len(bboxes) > 0 else np.empty((0, 5), dtype=np.int32)

        return bboxes_array

    def _extract_ins_segmentation_annotations(self, roi_masks_df, mask_col: str, cat_col: str, image_array: np.ndarray):
        """Extracts instance segmentation masks from ROI masks.

        TODO: Implement instance segmentation mask generation from individual masks.
        Returns tuple of (instance_mask_array, category_array)
        """
        raise NotImplementedError('Instance segmentation annotation extraction not yet implemented')

    def _extract_raycast_annotations(
        self, roi_masks_df, mask_col: str, cat_col: str, image_array: np.ndarray
    ) -> np.ndarray:
        """Extracts raycast annotations from per-cell binary mask contours.

        Uses CHAIN_APPROX_NONE (not SIMPLE) per §12.3 NOTE-01 — SIMPLE
        suppresses collinear edge pixels, producing sparser Shapely geometries
        and causing ray misses on small circular cells.

        Returns an array of shape (N, 35) float32 in unified format:
            [class_id, cx, cy, d_1, ..., d_32] — pixel space.
        """
        annotations = []

        if roi_masks_df is not None:
            for mask_row in roi_masks_df.iter_rows(named=True):
                mask_struct = mask_row[mask_col]
                mask_bytes = mask_struct['bytes'] if isinstance(mask_struct, dict) else mask_struct

                category = mask_row[cat_col]
                mask_array = self._decode_image(mask_bytes, is_mask=True)

                if mask_array.ndim > 2:
                    mask_array = mask_array[:, :, 0]

                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if not contours:
                    continue

                # Use largest contour only; ignore spurious noise contours
                contour = max(contours, key=cv2.contourArea)

                if len(contour) < 3:
                    continue

                # Contour shape is (M, 1, 2) — squeeze to (M, 2)
                pts = contour.squeeze(axis=1)
                coords = [(float(x), float(y)) for x, y in pts]
                poly = Polygon(coords)

                class_id = self.standardize_label(category)
                ann = polygon_to_raycast(poly, class_id)
                if ann is not None:
                    annotations.append(ann)

        if annotations:
            return np.stack(annotations).astype(np.float32)
        return np.zeros((0, 35), dtype=np.float32)
