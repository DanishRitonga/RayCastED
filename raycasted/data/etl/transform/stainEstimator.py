import numpy as np


class StainEstimator:
    @staticmethod
    def get_profile(image: np.ndarray, method: str = 'macenko') -> tuple[np.ndarray, np.ndarray]:
        """Routes the image to the requested stain estimation algorithm.
        Returns: (Stain_Matrix (2x3), Max_Concentrations (1x2)) or (None, None) if math fails.
        """
        if method.lower() == 'macenko':
            return StainEstimator._estimate_macenko(image)
        elif method.lower() == 'vahadane':
            raise NotImplementedError('Vahadane requires SPAMS library / Dictionary Learning.')
        elif method.lower() == 'reinhard':
            raise NotImplementedError('Reinhard uses LAB space mean/std, not OD vectors.')
        else:
            raise ValueError(f'Unknown normalization method: {method}')

    @staticmethod
    def _estimate_macenko(image: np.ndarray, Io: int = 240, alpha: float = 1, beta: float = 0.15) -> tuple:
        """Highly optimized NumPy implementation of the Macenko algorithm.
        Io: Transmitted light intensity (usually 240 or 255)
        alpha: Tolerance for the pseudo-min and pseudo-max (percentile)
        beta: OD threshold for transparent pixels
        """
        # 1. Convert RGB to Optical Density (OD) space
        # Reshape to (N, 3) where N is number of pixels
        img_reshaped = image.reshape((-1, 3)).astype(np.float64)

        # Calculate OD: OD = -log10(I / Io). Add 1 to avoid log(0)
        OD = -np.log10((img_reshaped + 1) / Io)

        # 2. Filter out background (transparent) pixels
        # A pixel is background if its OD in all channels is less than beta
        OD_hat = OD[~np.any(beta > OD, axis=1)]

        # Fail-safe: If the image is entirely white/empty, SVD collapses
        if len(OD_hat) < 100:
            return None, None

        # 3. Calculate Singular Value Decomposition (SVD)
        # Covariance matrix of OD_hat
        cov = np.cov(OD_hat, rowvar=False)

        # Eigenvalues and Eigenvectors (V is the principle directions)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Select the two largest eigenvectors (corresponding to H and E)
        eigvecs = eigvecs[:, [1, 2]]

        # 4. Project OD pixels onto the 2D plane defined by eigenvectors
        T_hat = np.dot(OD_hat, eigvecs)

        # Calculate the angle of each pixel in this 2D plane
        phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])

        # 5. Find robust extremes (Pseudo-min and Pseudo-max angles)
        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)

        # Convert the extreme angles back to OD space to get the stain vectors
        v_min = np.dot(eigvecs, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v_max = np.dot(eigvecs, np.array([np.cos(max_phi), np.sin(max_phi)]))

        # 6. Order the vectors (Heuristic: Hematoxylin has a larger Optical Density in the Red channel than Eosin)
        if v_min[0] > v_max[0]:
            HE = np.array([v_min, v_max])
        else:
            HE = np.array([v_max, v_min])

        # Normalize the vectors to unit length
        HE_normalized = HE / np.linalg.norm(HE, axis=1)[:, None]

        # 7. Calculate concentrations for this specific image
        # C = OD * pseudo_inverse(HE)
        C = np.dot(OD_hat, np.linalg.pinv(HE_normalized))

        # Find the 99th percentile of concentrations (robust maximum)
        max_C = np.percentile(C, 99, axis=0)

        return HE_normalized, max_C
