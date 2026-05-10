import cv2
import numpy as np
from scipy.ndimage import find_objects

from .._base import BaseDataIngestor
from ..file_handlers import ImageHandler, MatHandler, RayCastGPU


class MatInstParser(BaseDataIngestor):
    """Parser for .mat instance-map datasets (e.g., PanNuke-style).

    Source format is instance masks (inst_map), not polygon coordinates.
    Raycast extraction follows the same pattern as ParquetParser:
    binary mask → cv2.findContours → RayCastGPU.batch_polygon_to_raycast().
    """

    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, int]:
        """Process a single ROI: load image + .mat, route by annotation_type."""
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        image_array = ImageHandler.load_rgb(image_path)
        mat_data = MatHandler.load_mat(mask_path)

        if self.annotation_type == 'bbox':
            annotations_array = self._extract_bbox_annotations(mat_data, image_array)
            cat_array = None
        elif self.annotation_type == 'instance_mask':
            annotations_array, cat_array = self._extract_ins_segmentation_annotations(mat_data, image_array)
        elif self.annotation_type == 'raycast':
            annotations_array = self._extract_raycast_annotations(mat_data, image_array)
            cat_array = None
        else:
            raise ValueError(f'Unsupported annotation_type: {self.annotation_type}')

        tissue_origin = self.resolve_tissue()
        image_array, annotations_array = self.standardize_mpp(image_array, annotations_array)

        if cat_array is not None:
            return (roi_id, image_array, annotations_array, cat_array, tissue_origin)
        return (roi_id, image_array, annotations_array, tissue_origin)

    def _extract_bbox_annotations(self, mat_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extract bounding boxes from instance map using O(1) slice lookups.

        Returns an array of shape (N, 5) where each row is [xmin, ymin, xmax, ymax, class_id].
        """
        h, w = image_array.shape[:2]

        if 'inst_map' not in mat_data:
            raise KeyError("'inst_map' key not found in .mat file")

        instance_matrix = mat_data['inst_map'].astype(np.int32)

        if 'inst_type' not in mat_data:
            raise KeyError("'inst_type' key not found in .mat file")

        raw_types = mat_data['inst_type'].flatten()

        cats = [0]
        for raw_cat in raw_types:
            cats.append(self.standardize_label(raw_cat))

        slices = find_objects(instance_matrix)
        bboxes = []

        for i, slc in enumerate(slices):
            if slc is None:
                continue

            instance_id = i + 1
            if instance_id >= len(cats):
                continue

            ymin, ymax = slc[0].start, slc[0].stop
            xmin, xmax = slc[1].start, slc[1].stop

            class_id = cats[instance_id]
            bboxes.append([xmin, ymin, xmax, ymax, class_id])

        if len(bboxes) > 0:
            bboxes_array = np.array(bboxes, dtype=np.int32)
            bboxes_array[:, [0, 2]] = np.clip(bboxes_array[:, [0, 2]], 0, w)
            bboxes_array[:, [1, 3]] = np.clip(bboxes_array[:, [1, 3]], 0, h)
            valid_boxes = (bboxes_array[:, 2] > bboxes_array[:, 0]) & (bboxes_array[:, 3] > bboxes_array[:, 1])
            bboxes_array = bboxes_array[valid_boxes]
        else:
            bboxes_array = np.empty((0, 5), dtype=np.int32)

        return bboxes_array

    def _extract_ins_segmentation_annotations(self, mat_data: dict, image_array: np.ndarray):
        """Extract instance segmentation masks from .mat file.

        Returns tuple of (instance_mask_array, category_array).
        """
        raise NotImplementedError('Instance segmentation annotation extraction not yet implemented')

    def _extract_raycast_annotations(self, mat_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extract raycast annotations from .mat instance map.

        For each instance ID, extract binary mask → cv2.findContours → collect
        vertices → RayCastGPU.batch_polygon_to_raycast().
        """
        if 'inst_map' not in mat_data:
            raise KeyError("'inst_map' key not found in .mat file")

        instance_matrix = mat_data['inst_map'].astype(np.int32)

        if 'inst_type' not in mat_data:
            raise KeyError("'inst_type' key not found in .mat file")

        raw_types = mat_data['inst_type'].flatten()

        cats = [0]
        for raw_cat in raw_types:
            cats.append(self.standardize_label(raw_cat))

        slices = find_objects(instance_matrix)
        vertices = []
        class_ids = []

        for i, slc in enumerate(slices):
            if slc is None:
                continue

            instance_id = i + 1
            if instance_id >= len(cats):
                continue

            binary_mask = (instance_matrix[slc] == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue

            pts = contour.squeeze(axis=1).astype(np.float64)
            pts[:, 0] += slc[1].start
            pts[:, 1] += slc[0].start

            class_id = cats[instance_id]
            vertices.append(pts)
            class_ids.append(class_id)

        if vertices:
            return RayCastGPU.batch_polygon_to_raycast(
                vertices, np.array(class_ids, dtype=np.int64), n_rays=self.n_rays
            )
        return np.zeros((0, 3 + self.n_rays), dtype=np.float32)
