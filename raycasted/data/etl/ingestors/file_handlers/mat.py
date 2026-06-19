from scipy.io import loadmat


class MatHandler:
    """Stateless toolkit for MATLAB .mat file I/O."""

    @staticmethod
    def load_mat(path: str) -> dict:
        """Load a .mat file via scipy.io.loadmat."""
        return loadmat(path)
