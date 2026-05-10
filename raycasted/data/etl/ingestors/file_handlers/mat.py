import numpy as np
from scipy.io import loadmat
from scipy.ndimage import find_objects


class MatHandler:
    """Stateless toolkit for MATLAB .mat file I/O and instance map extraction."""

    @staticmethod
    def load_mat(path: str) -> dict:
        """Load a .mat file via scipy.io.loadmat."""
        return loadmat(path)

    @staticmethod
    def extract_instance_map(mat_data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract instance map and raw type vector from loaded .mat data."""
        if 'inst_map' not in mat_data:
            raise KeyError("'inst_map' key not found in .mat file")
        inst_map = mat_data['inst_map'].astype(np.int32)

        if 'inst_type' not in mat_data:
            raise KeyError("'inst_type' key not found in .mat file")
        raw_types = mat_data['inst_type'].flatten()

        return inst_map, raw_types

    @staticmethod
    def find_instance_slices(instance_matrix: np.ndarray) -> list:
        """Find bounding-box slices for each instance in the instance map."""
        return find_objects(instance_matrix)
