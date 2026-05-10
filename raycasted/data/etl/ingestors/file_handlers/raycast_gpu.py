import collections
import warnings

import numpy as np

from ...ops import polygon_to_raycast


class RayCastGPU:
    """Stateless toolkit for GPU-batched analytical ray-edge intersection.

    Uses lazy ``import torch`` to avoid hard-dependency on PyTorch during ETL.
    Falls back to shapely-based serial conversion when CUDA is unavailable.
    """

    @staticmethod
    def batch_polygon_to_raycast(
        vertices: list[np.ndarray],
        class_ids: np.ndarray,
        n_rays: int = 32,
        device: str = 'cuda',
        fallback_counter: collections.Counter | None = None,
    ) -> np.ndarray:
        """Batch-analytical polygon-to-raycast conversion on GPU.

        Args:
            vertices: List of (K_i, 2) numpy arrays — variable-length polygon vertex lists.
            class_ids: (N,) integer class labels per polygon.
            n_rays: Number of radial rays (default 32).
            device: Torch device string (default 'cuda').
            fallback_counter: Optional counter for diagnostic tracking.

        Returns:
            annotations: (N, 3 + n_rays) float32 in unified format
                [class_id, cx, cy, d_1, ..., d_R] — pixel space.
        """
        if not vertices:
            return np.zeros((0, 3 + n_rays), dtype=np.float32)

        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError('No CUDA device available')
        except (ImportError, RuntimeError) as exc:
            warnings.warn(f'GPU unavailable ({exc}), falling back to shapely', stacklevel=2)
            return RayCastGPU._fallback_shapely(vertices, class_ids, n_rays, fallback_counter)

        import torch

        n_polys = len(vertices)
        k_max = max(len(v) for v in vertices)

        v_padded, mask = RayCastGPU._pad_vertices(vertices, k_max, device, torch)
        directions = RayCastGPU._build_ray_directions(n_rays, device, torch)
        centroids = RayCastGPU._compute_centroids(v_padded, mask, k_max, torch)
        inside = RayCastGPU._point_in_polygon(centroids, v_padded, mask, k_max, torch)
        centroids = RayCastGPU._fallback_centroid(centroids, inside, v_padded, mask, torch)
        distances = RayCastGPU._solve_ray_intersections(centroids, v_padded, mask, directions, k_max, torch)

        return RayCastGPU._assemble_output(distances, centroids, class_ids, n_rays, n_polys, torch)

    @staticmethod
    def _fallback_shapely(
        vertices: list[np.ndarray],
        class_ids: np.ndarray,
        n_rays: int,
        fallback_counter: collections.Counter | None = None,
    ) -> np.ndarray:
        """Serial shapely fallback when GPU is unavailable."""
        from shapely.geometry import Polygon

        annotations = []
        for verts, cid in zip(vertices, class_ids):
            if len(verts) < 3:
                continue
            poly = Polygon(verts.tolist())
            ann = polygon_to_raycast(poly, int(cid), n_rays=n_rays, fallback_counter=fallback_counter)
            if ann is not None:
                annotations.append(ann)

        if annotations:
            return np.stack(annotations).astype(np.float32)
        return np.zeros((0, 3 + n_rays), dtype=np.float32)

    @staticmethod
    def _pad_vertices(vertices: list[np.ndarray], k_max: int, device: str, torch) -> tuple:
        """Pad variable-length polygons to k_max by repeating the last vertex."""
        n_polys = len(vertices)
        v_padded = torch.zeros(n_polys, k_max, 2, dtype=torch.float32, device=device)
        mask = torch.zeros(n_polys, k_max, dtype=torch.bool, device=device)

        for i, verts in enumerate(vertices):
            k = len(verts)
            v_padded[i, :k] = torch.from_numpy(verts.astype(np.float32))
            v_padded[i, k:] = torch.from_numpy(verts[-1:].astype(np.float32))
            mask[i, :k] = True

        return v_padded, mask

    @staticmethod
    def _build_ray_directions(n_rays: int, device: str, torch) -> tuple:
        """Compute unit direction vectors for each ray angle."""
        angles = torch.linspace(0, 2 * torch.pi, n_rays + 1, device=device)[:n_rays]
        return (torch.cos(angles), torch.sin(angles))

    @staticmethod
    def _compute_centroids(v_padded, mask, k_max, torch):
        """Area-weighted centroid via shoelace formula, vectorized over all polygons."""
        n_polys, n_verts, _ = v_padded.shape
        n_real = mask.sum(dim=1).long()
        poly_idx = torch.arange(n_polys, device=v_padded.device)

        v_next = torch.empty_like(v_padded)
        v_next[:, :-1, :] = v_padded[:, 1:, :]
        close_idx = n_real - 1
        v_next[poly_idx, close_idx] = v_padded[poly_idx, 0]

        last_edge = torch.arange(n_verts, device=v_padded.device).unsqueeze(0) == close_idx.unsqueeze(1)
        edge_mask = mask & (torch.arange(n_verts, device=v_padded.device).unsqueeze(0) < close_idx.unsqueeze(1))
        edge_mask = edge_mask | last_edge

        cross = v_padded[:, :, 0] * v_next[:, :, 1] - v_next[:, :, 0] * v_padded[:, :, 1]
        cross = cross * edge_mask.float()
        area = 0.5 * cross.sum(dim=1)

        cx = ((v_padded[:, :, 0] + v_next[:, :, 0]) * cross * edge_mask.float()).sum(dim=1)
        cy = ((v_padded[:, :, 1] + v_next[:, :, 1]) * cross * edge_mask.float()).sum(dim=1)

        safe_area = torch.where(area.abs() > 1e-12, area, torch.ones_like(area))
        cx = cx / (6.0 * safe_area)
        cy = cy / (6.0 * safe_area)

        degenerate = area.abs() < 1e-12
        cx = torch.where(degenerate, v_padded[:, 0, 0], cx)
        cy = torch.where(degenerate, v_padded[:, 0, 1], cy)

        return torch.stack([cx, cy], dim=-1)

    @staticmethod
    def _point_in_polygon(centroids, v_padded, mask, k_max, torch):
        """Crossing-number test to detect centroids outside their polygon."""
        n_polys, n_verts, _ = v_padded.shape
        n_real = mask.sum(dim=1).long()
        poly_idx = torch.arange(n_polys, device=centroids.device)

        v_next = torch.empty_like(v_padded)
        v_next[:, :-1, :] = v_padded[:, 1:, :]
        close_idx = n_real - 1
        v_next[poly_idx, close_idx] = v_padded[poly_idx, 0]

        last_edge = torch.arange(n_verts, device=centroids.device).unsqueeze(0) == close_idx.unsqueeze(1)
        edge_mask = mask & (torch.arange(n_verts, device=centroids.device).unsqueeze(0) < close_idx.unsqueeze(1))
        edge_mask = edge_mask | last_edge

        y_cond = (v_padded[:, :, 1] > centroids[:, 1:2]) != (v_next[:, :, 1] > centroids[:, 1:2])
        slope = (v_next[:, :, 0] - v_padded[:, :, 0]) * (centroids[:, 1:2] - v_padded[:, :, 1])
        denom = v_next[:, :, 1] - v_padded[:, :, 1]
        safe_denom = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
        x_intersect = v_padded[:, :, 0] + slope / safe_denom
        cross_right = x_intersect > centroids[:, 0:1]

        crossings = (y_cond & cross_right & edge_mask).sum(dim=1)
        inside = crossings % 2 == 1

        return inside

    @staticmethod
    def _fallback_centroid(centroids, inside, v_padded, mask, torch):
        """Move exterior centroids inside via binary search toward centroid of 3 nearest vertices."""
        outside = ~inside
        if not outside.any():
            return centroids

        n_polys = v_padded.shape[0]
        k_max = v_padded.shape[1]
        poly_idx = torch.arange(n_polys, device=centroids.device)

        diff = v_padded - centroids.unsqueeze(1)
        dist_sq = (diff * diff).sum(dim=-1)
        dist_sq = dist_sq.masked_fill(~mask, float('inf'))

        n_nearest = min(3, k_max)
        _, topk_idx = dist_sq.topk(n_nearest, dim=1, largest=False)

        topk_verts = v_padded[poly_idx.unsqueeze(1), topk_idx]
        target = topk_verts.mean(dim=1)

        candidates = centroids.clone()
        for frac in [0.5, 0.75, 0.875, 0.9375, 0.96875]:
            mid = centroids + frac * (target - centroids)
            mid_inside = RayCastGPU._point_in_polygon(mid, v_padded, mask, k_max, torch)
            candidates = torch.where(mid_inside.unsqueeze(1), mid, candidates)

        centroids = torch.where(outside.unsqueeze(1), candidates, centroids)

        return centroids

    @staticmethod
    def _solve_ray_intersections(centroids, v_padded, mask, directions, k_max, torch):
        """Cramer's rule batch solve for all (polygon, ray, edge) triples."""
        cos_d, sin_d = directions
        n_polys, n_verts, _ = v_padded.shape
        poly_idx = torch.arange(n_polys, device=centroids.device)
        n_real = mask.sum(dim=1).long()

        vjn = torch.empty_like(v_padded)
        vjn[:, :-1, :] = v_padded[:, 1:, :]
        vjn[poly_idx, n_real - 1] = v_padded[poly_idx, 0]

        last_edge = torch.arange(n_verts, device=centroids.device).unsqueeze(0) == (n_real - 1).unsqueeze(1)
        edge_mask = mask & (torch.arange(n_verts, device=centroids.device).unsqueeze(0) < n_real.unsqueeze(1))
        edge_mask = edge_mask | last_edge

        v1x = v_padded[:, :, 0].unsqueeze(1)
        v1y = v_padded[:, :, 1].unsqueeze(1)
        v2x = vjn[:, :, 0].unsqueeze(1)
        v2y = vjn[:, :, 1].unsqueeze(1)
        cx = centroids[:, 0].unsqueeze(1).unsqueeze(2)
        cy = centroids[:, 1].unsqueeze(1).unsqueeze(2)

        dx = v2x - v1x
        dy = v2y - v1y
        cos_exp = cos_d.view(1, -1, 1)
        sin_exp = sin_d.view(1, -1, 1)

        det = -cos_exp * dy + sin_exp * dx
        det_valid = det.abs() > 1e-10

        t_num = (v1x - cx) * (-dy) + (v1y - cy) * dx
        u_num = (v1x - cx) * (-sin_exp) + (v1y - cy) * cos_exp

        safe_det = torch.where(det_valid, det, torch.ones_like(det))
        t = t_num / safe_det
        u = u_num / safe_det

        valid = det_valid & (t > 1e-6) & (u >= 0) & (u <= 1)
        valid = valid & edge_mask.unsqueeze(1)

        dists = torch.where(valid, t, torch.full_like(t, 1e10))
        min_dists, _ = dists.min(dim=-1)

        return min_dists

    @staticmethod
    def _assemble_output(distances, centroids, class_ids, n_rays, n_polys, torch):
        """Assemble final (N, 3+n_rays) annotation array."""
        out = torch.zeros(n_polys, 3 + n_rays, dtype=torch.float32, device=distances.device)
        out[:, 0] = torch.from_numpy(class_ids.astype(np.float32)).to(distances.device)
        out[:, 1] = centroids[:, 0]
        out[:, 2] = centroids[:, 1]
        hit = distances < 1e9
        out[:, 3:] = torch.where(hit, distances, torch.zeros_like(distances))

        return out.cpu().numpy()
