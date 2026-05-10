import warnings

import numpy as np


class StainEstimatorGPU:
    """GPU-accelerated Macenko stain estimation using PyTorch.

    Uses lazy ``import torch`` to avoid hard-dependency on PyTorch during ETL.
    Falls back to NumPy-based :class:`StainEstimator` when CUDA is unavailable.
    """

    @staticmethod
    def get_profile(
        image: np.ndarray,
        method: str = 'macenko',
        device: str = 'cuda',
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Route to Macenko GPU estimation. Returns (stain_matrix, max_concentrations) or (None, None)."""
        if method.lower() != 'macenko':
            raise ValueError(f'GPU estimator only supports macenko, got {method}')

        return StainEstimatorGPU._estimate_macenko(image, device=device)

    @staticmethod
    def _estimate_macenko(
        image: np.ndarray,
        Io: int = 240,
        alpha: float = 1,
        beta: float = 0.15,
        device: str = 'cuda',
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError('No CUDA device available')
        except (ImportError, RuntimeError) as exc:
            warnings.warn(f'GPU unavailable ({exc}), falling back to NumPy', stacklevel=2)
            from .stainEstimator import StainEstimator

            return StainEstimator._estimate_macenko(image, Io=Io, alpha=alpha, beta=beta)

        import torch

        dev = torch.device(device)

        img_flat = image.reshape(-1, 3).astype(np.float64)
        t_img = torch.from_numpy(img_flat).to(device=dev, dtype=torch.float64)

        t_od = -torch.log10((t_img + 1.0) / Io)

        background_mask = (t_od < beta).any(dim=1)
        t_od_hat = t_od[~background_mask]

        if t_od_hat.shape[0] < 100:
            return None, None

        centered = t_od_hat - t_od_hat.mean(dim=0, keepdim=True)
        n = centered.shape[0]
        cov = (centered.T @ centered) / (n - 1)

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvecs = eigvecs[:, [1, 2]]

        t_hat = t_od_hat @ eigvecs
        phi = torch.atan2(t_hat[:, 1], t_hat[:, 0])

        alpha_q = alpha / 100.0
        min_phi = torch.quantile(phi, alpha_q)
        max_phi = torch.quantile(phi, 1.0 - alpha_q)

        v_min = eigvecs @ torch.tensor([torch.cos(min_phi), torch.sin(min_phi)], dtype=torch.float64, device=dev)
        v_max = eigvecs @ torch.tensor([torch.cos(max_phi), torch.sin(max_phi)], dtype=torch.float64, device=dev)

        he = torch.stack([v_min, v_max]) if v_min[0] > v_max[0] else torch.stack([v_max, v_min])

        he_normalized = he / he.norm(dim=1, keepdim=True)

        c = t_od_hat @ torch.linalg.pinv(he_normalized)
        max_c = torch.quantile(c, 0.99, dim=0)

        return he_normalized.cpu().numpy(), max_c.cpu().numpy()
