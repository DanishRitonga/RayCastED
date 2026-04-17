import multiprocessing
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import polars as pl
from shapely.geometry import Polygon

from ..ops import polygon_to_raycast
from ._base import BaseDataIngestor


def _decode_image(byte_string: bytes, is_mask: bool = False) -> np.ndarray:
    """Convert raw bytes to numpy array using OpenCV (module-level for pickling)."""
    np_arr = np.frombuffer(byte_string, np.uint8)
    flags = cv2.IMREAD_UNCHANGED if is_mask else cv2.IMREAD_COLOR
    decoded_img = cv2.imdecode(np_arr, flags)
    if decoded_img is None:
        raise ValueError('OpenCV failed to decode the byte array.')
    if not is_mask and len(decoded_img.shape) == 3:
        decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    return decoded_img


def _extract_raycast_annotations(
    roi_masks_df, mask_col: str, cat_col: str, namespace_map: dict, global_cell_map: dict
) -> np.ndarray:
    """Extract raycast annotations from per-cell binary mask contours (module-level for pickling)."""
    annotations = []

    if roi_masks_df is not None:
        for mask_row in roi_masks_df.iter_rows(named=True):
            mask_struct = mask_row[mask_col]
            mask_bytes = mask_struct['bytes'] if isinstance(mask_struct, dict) else mask_struct
            category = mask_row[cat_col]
            mask_array = _decode_image(mask_bytes, is_mask=True)

            if mask_array.ndim > 2:
                mask_array = mask_array[:, :, 0]

            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue

            pts = contour.squeeze(axis=1)
            coords = [(float(x), float(y)) for x, y in pts]
            poly = Polygon(coords)

            raw_str = str(category)
            standard_str = namespace_map[raw_str]
            class_id = global_cell_map[standard_str]

            ann = polygon_to_raycast(poly, class_id)
            if ann is not None:
                annotations.append(ann)

    if annotations:
        return np.stack(annotations).astype(np.float32)
    from raycasted.data.etl.utils import constants as _c

    return np.zeros((0, 3 + _c.N_RAYS), dtype=np.float32)


def _extract_bbox_annotations(
    roi_masks_df, mask_col: str, cat_col: str, namespace_map: dict, global_cell_map: dict
) -> np.ndarray:
    """Extract bounding box annotations from mask contours (module-level for pickling)."""
    bboxes = []

    if roi_masks_df is not None:
        for mask_row in roi_masks_df.iter_rows(named=True):
            mask_struct = mask_row[mask_col]
            mask_bytes = mask_struct['bytes'] if isinstance(mask_struct, dict) else mask_struct
            category = mask_row[cat_col]
            mask_array = _decode_image(mask_bytes, is_mask=True)

            if mask_array.ndim > 2:
                mask_array = mask_array[:, :, 0]

            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                raw_str = str(category)
                standard_str = namespace_map[raw_str]
                class_id = global_cell_map[standard_str]
                bboxes.append([class_id, x, y, x + w, y + h])

    return np.array(bboxes, dtype=np.int32) if bboxes else np.empty((0, 5), dtype=np.int32)


def _standardize_mpp(
    image: np.ndarray, annotations: np.ndarray, scale_factor: float, annotation_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Scale image and annotations by MPP factor (module-level for pickling)."""
    if scale_factor == 1.0:
        return image, annotations

    h, w = image.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if len(annotations) == 0:
        return scaled_image, annotations

    if annotation_type == 'bbox':
        scaled = annotations.copy()
        scaled[:, :4] = np.round(scaled[:, :4] * scale_factor).astype(np.int32)
        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, new_w)
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, new_h)
        return scaled_image, scaled
    elif annotation_type == 'raycast':
        scaled = annotations.copy()
        scaled[:, 1:] = scaled[:, 1:] * scale_factor
        return scaled_image, scaled

    return scaled_image, annotations


def _process_roi_worker(task: dict) -> tuple[str, np.ndarray, np.ndarray, int] | None:
    """Process a single ROI in a worker subprocess.

    All config needed for label/tissue resolution is passed in the task dict.
    """
    # Configure ray count for this worker (spawn doesn't inherit globals)
    n_rays = task.get('n_rays', 32)
    from raycasted.data.etl.utils.constants import configure_rays

    configure_rays(n_rays)

    try:
        rgb_bytes = task['rgb_bytes']
        image_array = _decode_image(rgb_bytes, is_mask=False)

        roi_id = task['roi_id']
        masks_df = task['masks_df']
        mask_col = task['mask_col']
        cat_col = task['cat_col']
        annotation_type = task['annotation_type']

        if annotation_type == 'bbox':
            annotations = _extract_bbox_annotations(
                masks_df, mask_col, cat_col, task['namespace_map'], task['global_cell_map']
            )
        elif annotation_type == 'raycast':
            annotations = _extract_raycast_annotations(
                masks_df, mask_col, cat_col, task['namespace_map'], task['global_cell_map']
            )
        else:
            raise ValueError(f'Unsupported annotation_type: {annotation_type}')

        image_array, annotations = _standardize_mpp(image_array, annotations, task['scale_factor'], annotation_type)

        return (roi_id, image_array, annotations, task['tissue_origin'])
    except Exception as e:
        print(f'  Worker error on {task.get("roi_id", "?")}: {e}')
        return None


class ParquetIngestor(BaseDataIngestor):  # noqa: D101
    def __init__(self, config: dict, workers: int = 1):
        self._workers = workers
        super().__init__(config)

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

        # Read parquet into memory in the main process (fast with Polars)
        df_rgb = lf.select(['internal_roi_id', rgb_col, tissue_col]).collect()
        df_masks = lf.select(['internal_roi_id', mask_col, cat_col]).explode([mask_col, cat_col]).drop_nulls().collect()

        masks_by_roi = {}
        if not df_masks.is_empty():
            raw_dict = df_masks.partition_by('internal_roi_id', as_dict=True)
            masks_by_roi = {k[0]: v for k, v in raw_dict.items()}

        # Build task list — each task is a self-contained dict with all data needed
        tasks = []
        for rgb_row in df_rgb.iter_rows(named=True):
            internal_id = rgb_row['internal_roi_id']
            rgb_struct = rgb_row[rgb_col]
            rgb_bytes = rgb_struct['bytes'] if isinstance(rgb_struct, dict) else rgb_struct
            raw_tissue_id = rgb_row[tissue_col]

            tasks.append(
                {
                    'roi_id': f'{base_roi_name}_roi_{internal_id}',
                    'rgb_bytes': rgb_bytes,
                    'masks_df': masks_by_roi.get(internal_id),
                    'mask_col': mask_col,
                    'cat_col': cat_col,
                    'annotation_type': self.annotation_type,
                    'namespace_map': self.namespace_map,
                    'global_cell_map': self.global_cell_map,
                    'tissue_origin': self.resolve_tissue(raw_tissue_id),
                    'scale_factor': self.scale_factor,
                    'n_rays': self.config.get('n_rays', 32),
                }
            )

        # Dispatch
        if self._workers <= 1 or len(tasks) <= 1:
            # Sequential path — avoids process pool overhead for small workloads
            for task in tasks:
                result = _process_roi_worker(task)
                if result is not None:
                    yield result
        else:
            n_workers = min(self._workers, len(tasks))
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(_process_roi_worker, t): t['roi_id'] for t in tasks}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        yield result

    def _identify_columns(self, schema: pl.Schema) -> tuple[str, str, str, str]:
        rgb_col, mask_col, cat_col, tissue_col = None, None, None, None

        for col_name, dtype in schema.items():
            if isinstance(dtype, pl.Struct) or dtype == pl.Binary:
                rgb_col = col_name
            elif isinstance(dtype, pl.List) and (isinstance(dtype.inner, pl.Struct) or dtype.inner == pl.Binary):
                mask_col = col_name
            elif isinstance(dtype, pl.List) and dtype.inner in [pl.Int64, pl.Int32, pl.UInt32, pl.Int8]:
                cat_col = col_name
            elif dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.Int8] and not isinstance(dtype, pl.List):
                tissue_col = col_name

        return rgb_col, mask_col, cat_col, tissue_col

    def _decode_image(self, byte_string: bytes, is_mask: bool = False) -> np.ndarray:
        """Helper to convert raw bytes back into numpy arrays using OpenCV."""
        return _decode_image(byte_string, is_mask)

    def _extract_bbox_annotations(
        self,
        roi_masks_df,
        mask_col: str,
        cat_col: str,
        image_array: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        return _extract_bbox_annotations(roi_masks_df, mask_col, cat_col, self.namespace_map, self.global_cell_map)

    def _extract_ins_segmentation_annotations(self, roi_masks_df, mask_col: str, cat_col: str, image_array: np.ndarray):
        raise NotImplementedError('Instance segmentation annotation extraction not yet implemented')

    def _extract_raycast_annotations(
        self,
        roi_masks_df,
        mask_col: str,
        cat_col: str,
        image_array: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        return _extract_raycast_annotations(roi_masks_df, mask_col, cat_col, self.namespace_map, self.global_cell_map)
