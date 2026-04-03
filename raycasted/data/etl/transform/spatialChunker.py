import math
from collections.abc import Generator
from typing import Any

import numpy as np


class SpatialChunker:
    """A memory-only functional transformer that chunks massive spatial arrays.
    It mathematically guarantees a minimum overlap to preserve boundary-straddling annotations.
    """

    def __init__(self, config: dict[str, Any]):
        self.max_size = config.get('max_size', 1024)
        self.annotation_type = config.get('annotation_type', 'bbox').lower()

        # Minimum area ratio (0.0 to 1.0) required to keep a sliced bounding box
        self.min_area_ratio = config.get('min_sliver_ratio', 0.20)

        # Fetch the overlap percentage from your global settings (e.g., 0.10 for 10%)
        self.overlap_pct = config.get('patching_overlap_pct', 0.10)

    def process_roi(
        self, roi_id: str, image: np.ndarray, annotations: Any, tissue: int
    ) -> Generator[tuple[str, np.ndarray, Any, int], None, None]:
        """Takes a single ROI's data in memory. If it's larger than max_size, it yields
        multiple overlapping chunks. If perfectly sized or smaller, yields the original.
        """
        h, w = image.shape[:2]

        if h <= self.max_size and w <= self.max_size:
            # The "Goldilocks" or "Needs Padding" zone. Yield as-is.
            yield (roi_id, image, annotations, tissue)
            return

        # It's too big. Generate overlapping steps.
        y_steps = self._get_dynamic_steps(h, self.max_size, self.overlap_pct)
        x_steps = self._get_dynamic_steps(w, self.max_size, self.overlap_pct)

        for y in y_steps:
            for x in x_steps:
                chunk_id = f'{roi_id}_x{x}_y{y}'

                # 1. Slice the RGB Image
                img_chunk = image[y : y + self.max_size, x : x + self.max_size]
                chunk_h, chunk_w = img_chunk.shape[:2]

                # 2. Slice and Filter Annotations
                ann_chunk = self._slice_annotations(annotations, x, y, chunk_w, chunk_h)

                yield (chunk_id, img_chunk, ann_chunk, tissue)

    def _get_dynamic_steps(self, length: int, chunk_size: int, min_overlap_pct: float) -> np.ndarray:
        """Calculates evenly distributed starting indices with a guaranteed minimum overlap."""
        if length <= chunk_size:
            return np.array([0])

        overlap_pixels = int(chunk_size * min_overlap_pct)
        effective_stride = chunk_size - overlap_pixels

        # Ensure stride is at least 1 to prevent infinite loops, though mathematically unlikely here
        effective_stride = max(1, effective_stride)

        # Calculate required number of chunks to cover the length with the effective stride
        n_chunks = 1 + math.ceil((length - chunk_size) / effective_stride)

        # linspace naturally distributes the overlap evenly across all seams
        steps = np.linspace(0, length - chunk_size, n_chunks, dtype=int)
        return steps

    def _slice_annotations(self, annotations: Any, x_start: int, y_start: int, chunk_w: int, chunk_h: int) -> Any:
        """Routes the slicing logic based on the annotation type."""
        if len(annotations) == 0:
            return annotations

        if self.annotation_type == 'bbox':
            return self._slice_bboxes(annotations, x_start, y_start, chunk_w, chunk_h)

        elif self.annotation_type == 'instance_mask':
            return annotations[y_start : y_start + chunk_h, x_start : x_start + chunk_w]

        else:
            raise NotImplementedError(f'Chunk slicing for {self.annotation_type} is not yet implemented.')

    def _slice_bboxes(self, bboxes: np.ndarray, x_start: int, y_start: int, chunk_w: int, chunk_h: int) -> np.ndarray:
        """Shifts, clips, and applies the Area Preservation Filter to bounding boxes."""
        # Calculate original areas before any clipping
        orig_w = bboxes[:, 2] - bboxes[:, 0]
        orig_h = bboxes[:, 3] - bboxes[:, 1]
        orig_areas = orig_w * orig_h

        # Create a copy and shift coordinates relative to the chunk's top-left corner
        shifted = bboxes.copy()
        shifted[:, 0] -= x_start  # xmin
        shifted[:, 2] -= x_start  # xmax
        shifted[:, 1] -= y_start  # ymin
        shifted[:, 3] -= y_start  # ymax

        # Clip boxes to the boundaries of the new chunk
        shifted[:, [0, 2]] = np.clip(shifted[:, [0, 2]], 0, chunk_w)
        shifted[:, [1, 3]] = np.clip(shifted[:, [1, 3]], 0, chunk_h)

        # Calculate new areas after clipping
        new_w = shifted[:, 2] - shifted[:, 0]
        new_h = shifted[:, 3] - shifted[:, 1]
        new_areas = new_w * new_h

        # FILTER 1: Must have positive width and height
        valid_dims = (new_w > 0) & (new_h > 0)

        # FILTER 2: Area preservation (drops "slivers" sitting on the chunk boundary)
        with np.errstate(divide='ignore', invalid='ignore'):
            area_ratio = np.where(orig_areas > 0, new_areas / orig_areas, 0)

        valid_area = area_ratio >= self.min_area_ratio

        # Combine masks and apply
        keep_mask = valid_dims & valid_area

        return shifted[keep_mask]
