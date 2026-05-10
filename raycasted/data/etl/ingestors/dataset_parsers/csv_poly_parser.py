import cv2
import numpy as np
import polars as pl

from ...utils.constants import N_RAYS
from .._base import BaseDataIngestor
from ..file_handlers import ImageHandler, RayCastGPU


class CSVPolyParser(BaseDataIngestor):
    """Parser for CSV files containing polygon coordinate columns."""

    def process_item(self, row: dict):
        """Process a single registry row into standardized annotation format."""
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        image_array = ImageHandler.load_rgb(image_path)

        try:
            df = pl.read_csv(mask_path)
        except Exception as e:
            raise ValueError(f'Failed to read CSV at {mask_path}: {e}')

        if self.annotation_type == 'bbox':
            annotations_array = self._extract_bbox_annotations(df, image_array)
            cat_array = None
        elif self.annotation_type == 'polygon':
            annotations_array = self._extract_polygon_annotations(df, image_array)
            cat_array = None
        elif self.annotation_type == 'raycast':
            annotations_array = self._extract_raycast_annotations(df, image_array)
            cat_array = None
        else:
            raise ValueError(f'Unsupported annotation_type: {self.annotation_type}')

        tissue_origin = self.resolve_tissue()
        image_array, annotations_array = self.standardize_mpp(image_array, annotations_array)

        if cat_array is not None:
            return (roi_id, image_array, annotations_array, cat_array, tissue_origin)
        return (roi_id, image_array, annotations_array, tissue_origin)

    def _extract_bbox_annotations(self, df: pl.DataFrame, image_array: np.ndarray) -> np.ndarray:
        """Extracts bounding boxes from polygon coordinates in CSV."""
        col_map = self.config.get('csv_column_map', {})
        col_x = col_map.get('x_coords')
        col_y = col_map.get('y_coords')
        col_cat = col_map.get('category')

        if not all([col_x, col_y, col_cat]):
            raise KeyError('Missing required csv_column_map keys. Need x_coords, y_coords, and category.')

        h, w = image_array.shape[:2]
        bboxes = []

        for cell_row in df.iter_rows(named=True):
            x_str = cell_row[col_x]
            y_str = cell_row[col_y]

            if not x_str or not y_str:
                continue

            x_arr = np.array(x_str.split(','), dtype=np.int32)
            y_arr = np.array(y_str.split(','), dtype=np.int32)
            pts = np.column_stack((x_arr, y_arr))

            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(pts)

            raw_category = cell_row[col_cat]
            standardized_category = self.standardize_label(raw_category)

            bboxes.append([standardized_category, x_bbox, y_bbox, x_bbox + w_bbox, y_bbox + h_bbox])

        if len(bboxes) > 0:
            bboxes_array = np.array(bboxes, dtype=np.int32)
            bboxes_array[:, [0, 2]] = np.clip(bboxes_array[:, [0, 2]], 0, w)
            bboxes_array[:, [1, 3]] = np.clip(bboxes_array[:, [1, 3]], 0, h)
            valid_boxes = (bboxes_array[:, 2] > bboxes_array[:, 0]) & (bboxes_array[:, 3] > bboxes_array[:, 1])
            bboxes_array = bboxes_array[valid_boxes]
        else:
            bboxes_array = np.empty((0, 5), dtype=np.int32)

        return bboxes_array

    def _extract_polygon_annotations(self, df: pl.DataFrame, image_array: np.ndarray) -> np.ndarray:
        """Extracts polygon coordinates from CSV."""
        raise NotImplementedError('Polygon annotation extraction not yet implemented')

    def _extract_ins_segmentation_annotations(self, df: pl.DataFrame, image_array: np.ndarray):
        """Extracts instance segmentation masks from CSV."""
        raise NotImplementedError('Instance segmentation annotation extraction not yet implemented')

    def _extract_raycast_annotations(self, df: pl.DataFrame, image_array: np.ndarray) -> np.ndarray:
        """Extracts raycast annotations from CSV polygon coordinates via GPU batch."""
        col_map = self.config.get('csv_column_map', {})
        col_x = col_map.get('x_coords')
        col_y = col_map.get('y_coords')
        col_cat = col_map.get('category')

        if not all([col_x, col_y, col_cat]):
            raise KeyError('Missing required csv_column_map keys. Need x_coords, y_coords, and category.')

        vertices = []
        class_ids = []

        for cell_row in df.iter_rows(named=True):
            x_str = cell_row[col_x]
            y_str = cell_row[col_y]

            if not x_str or not y_str:
                continue

            x_arr = np.array(x_str.split(','), dtype=np.float64)
            y_arr = np.array(y_str.split(','), dtype=np.float64)

            if len(x_arr) < 3:
                continue

            contour = np.column_stack((x_arr, y_arr)).astype(np.float64)

            raw_category = cell_row[col_cat]
            class_id = self.standardize_label(raw_category)

            vertices.append(contour)
            class_ids.append(class_id)

        if vertices:
            return RayCastGPU.batch_polygon_to_raycast(vertices, np.array(class_ids, dtype=np.int64))
        return np.zeros((0, 3 + N_RAYS), dtype=np.float32)
