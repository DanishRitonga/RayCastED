import cv2
import numpy as np
import polars as pl
from shapely.geometry import Polygon

from ..ops import polygon_to_raycast
from ._base import BaseDataIngestor


class CSVPolygonIngestor(BaseDataIngestor):
    def process_item(self, row: dict):
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        # 1. Load the RGB Image
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError(f'Failed to read image at {image_path}')
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # 2. Read the CSV using Polars
        try:
            df = pl.read_csv(mask_path)
        except Exception as e:
            raise ValueError(f'Failed to read CSV at {mask_path}: {e}')

        # 3. Route to appropriate annotation extractor based on annotation_type
        if self.annotation_type == 'bbox':
            annotations_array, cat_array = self._extract_bbox_annotations(df, image_array), None
        elif self.annotation_type == 'polygon':
            annotations_array, cat_array = self._extract_polygon_annotations(df, image_array), None
        elif self.annotation_type == 'raycast':
            annotations_array, cat_array = self._extract_raycast_annotations(df, image_array), None
        else:
            raise ValueError(f'Unsupported annotation_type: {self.annotation_type}')

        # 4. Apply common post-processing
        tissue_origin = self.resolve_tissue()
        image_array, annotations_array = self.standardize_mpp(image_array, annotations_array)

        # 5. Return based on annotation type
        if cat_array is not None:
            return (roi_id, image_array, annotations_array, cat_array, tissue_origin)
        else:
            return (roi_id, image_array, annotations_array, tissue_origin)

    def _extract_bbox_annotations(self, df: pl.DataFrame, image_array: np.ndarray) -> np.ndarray:
        """Extracts bounding boxes from polygon coordinates in CSV.

        Returns an array of shape (N, 5) where each row is [xmin, ymin, xmax, ymax, class_id]
        """
        # Fetch column mapping
        col_map = self.config.get('csv_column_map', {})
        col_x = col_map.get('x_coords')
        col_y = col_map.get('y_coords')
        col_cat = col_map.get('category')

        if not all([col_x, col_y, col_cat]):
            raise KeyError('Missing required csv_column_map keys. Need x_coords, y_coords, and category.')

        h, w = image_array.shape[:2]
        bboxes = []

        for cell_row in df.iter_rows(named=True):
            # Grab the comma-separated strings
            x_str = cell_row[col_x]
            y_str = cell_row[col_y]

            # Skip empty or malformed rows
            if not x_str or not y_str:
                continue

            # Split the strings and cast directly to an integer numpy array
            x_arr = np.array(x_str.split(','), dtype=np.int32)
            y_arr = np.array(y_str.split(','), dtype=np.int32)

            # Zip them together into the (N, 2) shape OpenCV expects
            pts = np.column_stack((x_arr, y_arr))

            # Extract bounding box directly from polygon coordinates
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(pts)

            # Extract and Standardize the Category
            raw_category = cell_row[col_cat]
            standardized_category = self.standardize_label(raw_category)

            bboxes.append([standardized_category, x_bbox, y_bbox, x_bbox + w_bbox, y_bbox + h_bbox])

        # Safe bounding box array initialization with boundary clipping and degenerate box filtering
        if len(bboxes) > 0:
            bboxes_array = np.array(bboxes, dtype=np.int32)

            # Clip X coordinates (xmin at index 0, xmax at index 2) to [0, w]
            bboxes_array[:, [0, 2]] = np.clip(bboxes_array[:, [0, 2]], 0, w)

            # Clip Y coordinates (ymin at index 1, ymax at index 3) to [0, h]
            bboxes_array[:, [1, 3]] = np.clip(bboxes_array[:, [1, 3]], 0, h)

            # Filter out degenerate boxes (where area became 0 after clipping)
            valid_boxes = (bboxes_array[:, 2] > bboxes_array[:, 0]) & (bboxes_array[:, 3] > bboxes_array[:, 1])
            bboxes_array = bboxes_array[valid_boxes]

        else:
            bboxes_array = np.empty((0, 5), dtype=np.int32)

        return bboxes_array

    def _extract_polygon_annotations(self, df: pl.DataFrame, image_array: np.ndarray) -> np.ndarray:
        """Extracts polygon coordinates from CSV.

        TODO: Implement polygon extraction from CSV coordinates.
        Returns an array of polygons with their class labels.
        """
        raise NotImplementedError('Polygon annotation extraction not yet implemented')

    def _extract_raycast_annotations(self, df: pl.DataFrame, image_array: np.ndarray) -> np.ndarray:
        """Extracts raycast annotations from CSV polygon coordinates.

        Returns an array of shape (N, 35) float32 in unified format:
            [class_id, cx, cy, d_1, ..., d_32] — pixel space.
        """
        col_map = self.config.get('csv_column_map', {})
        col_x = col_map.get('x_coords')
        col_y = col_map.get('y_coords')
        col_cat = col_map.get('category')

        if not all([col_x, col_y, col_cat]):
            raise KeyError('Missing required csv_column_map keys. Need x_coords, y_coords, and category.')

        annotations = []

        for cell_row in df.iter_rows(named=True):
            x_str = cell_row[col_x]
            y_str = cell_row[col_y]

            if not x_str or not y_str:
                continue

            x_arr = np.array(x_str.split(','), dtype=np.float64)
            y_arr = np.array(y_str.split(','), dtype=np.float64)

            if len(x_arr) < 3:
                continue

            coords = list(zip(x_arr, y_arr))
            poly = Polygon(coords)

            raw_category = cell_row[col_cat]
            class_id = self.standardize_label(raw_category)

            ann = polygon_to_raycast(poly, class_id)
            if ann is not None:
                annotations.append(ann)

        if annotations:
            return np.stack(annotations).astype(np.float32)
        return np.zeros((0, 35), dtype=np.float32)
