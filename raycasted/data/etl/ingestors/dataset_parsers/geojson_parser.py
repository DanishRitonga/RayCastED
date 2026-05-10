import cv2
import numpy as np

from .._base import BaseDataIngestor
from ..file_handlers import GeoJSONHandler, ImageHandler, RayCastGPU


class GeoJSONParser(BaseDataIngestor):
    """Parser for GeoJSON annotation files with polygon/bbox/raycast support."""

    def process_item(self, row: dict) -> tuple:
        """Process a single registry row (image + GeoJSON mask pair)."""
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        image_array = ImageHandler.load_rgb(image_path)
        geo_data = GeoJSONHandler.load_json(mask_path)

        if self.annotation_type == 'bbox':
            annotations_array = self._extract_bbox_annotations(geo_data, image_array)
            cat_array = None
        elif self.annotation_type == 'polygon':
            annotations_array = self._extract_polygon_annotations(geo_data, image_array)
            cat_array = None
        elif self.annotation_type == 'instance_mask':
            annotations_array, cat_array = self._extract_ins_segmentation_annotations(geo_data, image_array)
        elif self.annotation_type == 'raycast':
            annotations_array = self._extract_raycast_annotations(geo_data, image_array)
            cat_array = None
        else:
            raise ValueError(f'Unsupported annotation_type: {self.annotation_type}')

        tissue_origin = self.resolve_tissue()
        image_array, annotations_array = self.standardize_mpp(image_array, annotations_array)

        if cat_array is not None:
            return (roi_id, image_array, annotations_array, cat_array, tissue_origin)
        return (roi_id, image_array, annotations_array, tissue_origin)

    def _extract_category(self, properties: dict, default: str) -> str:
        """Extracts the exact classification name provided by the dataset authors."""
        if 'classification' in properties and 'name' in properties['classification']:
            return str(properties['classification']['name'])

        if 'classId' in properties:
            return str(properties['classId'])

        return default

    def _extract_bbox_annotations(self, geo_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extracts bounding boxes from GeoJSON polygon coordinates.

        Returns an array of shape (N, 5) where each row is [xmin, ymin, xmax, ymax, class_id]
        """
        features = geo_data.get('features', [])
        bboxes = []

        for feature in features:
            geom_type = feature.get('geometry', {}).get('type')

            if geom_type not in ['Polygon', 'MultiPolygon']:
                continue

            coordinates = feature['geometry']['coordinates']
            properties = feature.get('properties', {})

            raw_category = self._extract_category(properties, default='unlabeled')
            standardized_category = self.standardize_label(raw_category)

            if geom_type == 'Polygon':
                exterior_ring = coordinates[0]
                pts = np.array(exterior_ring, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                bboxes.append([x, y, x + w, y + h, standardized_category])

            elif geom_type == 'MultiPolygon':
                for poly_coords in coordinates:
                    exterior_ring = poly_coords[0]
                    pts = np.array(exterior_ring, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    bboxes.append([x, y, x + w, y + h, standardized_category])

        if len(bboxes) > 0:
            bboxes_array = np.array(bboxes, dtype=np.int32)
            h, w = image_array.shape[:2]
            bboxes_array[:, [0, 2]] = np.clip(bboxes_array[:, [0, 2]], 0, w)
            bboxes_array[:, [1, 3]] = np.clip(bboxes_array[:, [1, 3]], 0, h)
            valid_boxes = (bboxes_array[:, 2] > bboxes_array[:, 0]) & (bboxes_array[:, 3] > bboxes_array[:, 1])
            bboxes_array = bboxes_array[valid_boxes]
        else:
            bboxes_array = np.empty((0, 5), dtype=np.int32)

        return bboxes_array

    def _extract_polygon_annotations(self, geo_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extracts polygon coordinates from GeoJSON."""
        raise NotImplementedError('Polygon annotation extraction not yet implemented')

    def _extract_ins_segmentation_annotations(self, geo_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extracts segmentation masks from GeoJSON polygon coordinates."""
        raise NotImplementedError('Instance segmentation annotation extraction not yet implemented')

    def _extract_raycast_annotations(self, geo_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extracts raycast annotations from GeoJSON polygon coordinates.

        Returns an array of shape (N, 35) float32 in unified format:
            [class_id, cx, cy, d_1, ..., d_32] — pixel space.
        """
        features = geo_data.get('features', [])
        vertices = []
        class_ids = []

        for feature in features:
            geom_type = feature.get('geometry', {}).get('type')
            if geom_type not in ['Polygon', 'MultiPolygon']:
                continue

            coordinates = feature['geometry']['coordinates']
            properties = feature.get('properties', {})

            raw_category = self._extract_category(properties, default='unlabeled')
            class_id = self.standardize_label(raw_category)

            if geom_type == 'Polygon':
                pts = np.array(coordinates[0], dtype=np.float64)
                if len(pts) >= 3:
                    vertices.append(pts)
                    class_ids.append(class_id)

            elif geom_type == 'MultiPolygon':
                for poly_coords in coordinates:
                    pts = np.array(poly_coords[0], dtype=np.float64)
                    if len(pts) >= 3:
                        vertices.append(pts)
                        class_ids.append(class_id)

        if vertices:
            return RayCastGPU.batch_polygon_to_raycast(
                vertices, np.array(class_ids, dtype=np.int64), n_rays=self.n_rays
            )
        return np.zeros((0, 3 + self.n_rays), dtype=np.float32)
