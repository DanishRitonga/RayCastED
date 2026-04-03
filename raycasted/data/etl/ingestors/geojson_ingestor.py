import cv2
import numpy as np
import orjson
from shapely.geometry import MultiPolygon, Polygon

from ..ops import polygon_to_raycast
from ._base import BaseDataIngestor


class GeoJSONIngestor(BaseDataIngestor):
    def process_item(self, row: dict) -> tuple:
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        # 1. Load the RGB Image
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError(f'Failed to read image at {image_path}')
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # 2. Parse the GeoJSON using the fast Rust backend
        with open(mask_path, 'rb') as f:
            geo_data = orjson.loads(f.read())

        # 3. Route to appropriate annotation extractor based on annotation_type
        if self.annotation_type == 'bbox':
            annotations_array, cat_array = self._extract_bbox_annotations(geo_data, image_array), None
        elif self.annotation_type == 'polygon':
            annotations_array, cat_array = self._extract_polygon_annotations(geo_data, image_array), None
        elif self.annotation_type == 'instance_mask':
            annotations_array, cat_array = self._extract_ins_segmentation_annotations(geo_data, image_array)
        elif self.annotation_type == 'raycast':
            annotations_array, cat_array = self._extract_raycast_annotations(geo_data, image_array), None
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

            # --- THE ONTOLOGY GATEKEEPER ---
            # Extract the RAW string category from the dataset
            raw_category = self._extract_category(properties, default='unlabeled')

            # Instantly standardize it using the Base Class method
            standardized_category = self.standardize_label(raw_category)

            # Extract bounding boxes directly from polygon coordinates
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
                    bboxes.append([standardized_category, x, y, x + w, y + h])

        # 4. Safe bounding box array initialization to guarantee (N, 5) shape
        if len(bboxes) > 0:
            bboxes_array = np.array(bboxes, dtype=np.int32)

            # Extract image dimensions
            h, w = image_array.shape[:2]

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

    def _extract_polygon_annotations(self, geo_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extracts polygon coordinates from GeoJSON.

        TODO: Implement polygon extraction for 'Star-convex' and other polygon types.
        Returns an array of polygons with their class labels.
        """
        raise NotImplementedError('Polygon annotation extraction not yet implemented')

    def _extract_ins_segmentation_annotations(self, geo_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extracts segmentation masks from GeoJSON polygon coordinates.

        TODO: Implement segmentation mask generation from polygon boundaries.
        Returns a binary mask array of shape (H, W) or (H, W, num_classes).
        """
        raise NotImplementedError('Instance segmentation annotation extraction not yet implemented')

    def _extract_raycast_annotations(self, geo_data: dict, image_array: np.ndarray) -> np.ndarray:
        """Extracts raycast annotations from GeoJSON polygon coordinates.

        Returns an array of shape (N, 35) float32 in unified format:
            [class_id, cx, cy, d_1, ..., d_32] — pixel space.
        """
        features = geo_data.get('features', [])
        annotations = []

        for feature in features:
            geom_type = feature.get('geometry', {}).get('type')
            if geom_type not in ['Polygon', 'MultiPolygon']:
                continue

            coordinates = feature['geometry']['coordinates']
            properties = feature.get('properties', {})

            raw_category = self._extract_category(properties, default='unlabeled')
            class_id = self.standardize_label(raw_category)

            if geom_type == 'Polygon':
                poly = Polygon(coordinates[0])
                ann = polygon_to_raycast(poly, class_id)
                if ann is not None:
                    annotations.append(ann)

            elif geom_type == 'MultiPolygon':
                multi = MultiPolygon([Polygon(pc[0]) for pc in coordinates])
                for poly in multi.geoms:
                    ann = polygon_to_raycast(poly, class_id)
                    if ann is not None:
                        annotations.append(ann)

        if annotations:
            return np.stack(annotations).astype(np.float32)
        return np.zeros((0, 35), dtype=np.float32)
