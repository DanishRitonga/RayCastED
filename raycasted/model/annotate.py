"""RayCastED — Polygon Annotator (Phase 7).

RayCastAnnotator draws raycast polygon predictions on images:
    - Polygon outline (green by default)
    - Centroid marker (red dot)
    - Optional ray visualization for debugging

Uses OpenCV (cv2) for all drawing operations.
"""

import numpy as np

from raycasted.data.etl.ops.convert import decode_to_vertices
from raycasted.data.etl.utils import constants as _const


class RayCastAnnotator:
    """Draw raycast polygon detections on images using OpenCV.

    Args:
        im: Image to annotate (modified in-place).
        poly_color: BGR color for polygon outline.
        centroid_color: BGR color for centroid dot.
        ray_color: BGR color for ray lines (debug mode).
        line_width: Line thickness.
        centroid_radius: Radius of centroid dot.
        show_rays: Whether to draw individual rays.
    """

    def __init__(
        self,
        im: np.ndarray,
        poly_color: tuple[int, int, int] = (0, 255, 0),
        centroid_color: tuple[int, int, int] = (0, 0, 255),
        ray_color: tuple[int, int, int] = (255, 255, 0),
        line_width: int = 2,
        centroid_radius: int = 4,
        show_rays: bool = False,
    ):
        self.im = im
        self.poly_color = poly_color
        self.centroid_color = centroid_color
        self.ray_color = ray_color
        self.line_width = line_width
        self.centroid_radius = centroid_radius
        self.show_rays = show_rays

    def draw_polygons(self, polygons: np.ndarray) -> None:
        """Draw all polygon detections on the image.

        Args:
            polygons: [N, 36] array where each row is
                [cx, cy, d_1..d_32, score, cls_idx].
                Only first 34 columns are used for drawing.
        """
        import cv2

        if polygons is None or len(polygons) == 0:
            return

        polygons = np.asarray(polygons, dtype=np.float64)
        cx = polygons[:, 0]
        cy = polygons[:, 1]
        rays = polygons[:, 2 : 2 + _const.N_RAYS]

        # Decode to vertices: [N, n_rays, 2]
        vertices = decode_to_vertices(rays, cx, cy)

        for i in range(len(polygons)):
            # Draw polygon outline
            pts = vertices[i].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                self.im, [pts], isClosed=True, color=self.poly_color, thickness=self.line_width, lineType=cv2.LINE_AA
            )

            # Draw centroid
            cx_i, cy_i = int(cx[i]), int(cy[i])
            cv2.circle(self.im, (cx_i, cy_i), self.centroid_radius, self.centroid_color, -1, cv2.LINE_AA)

            # Optional: draw individual rays
            if self.show_rays:
                for j in range(_const.N_RAYS):
                    end_x = int(cx[i] + rays[i, j] * _const.RAY_COS[j])
                    end_y = int(cy[i] + rays[i, j] * _const.RAY_SIN[j])
                    cv2.line(self.im, (cx_i, cy_i), (end_x, end_y), self.ray_color, 1, cv2.LINE_AA)

    def result(self) -> np.ndarray:
        """Return the annotated image."""
        return self.im
