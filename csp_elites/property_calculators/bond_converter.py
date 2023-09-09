import torch
from megnet.data.graph import Converter


class BondConverterTorch(Converter):
    """
    This class is an exact copy of BondConverter converter from MEGNET except implemented in torch to allow
    automatic differentiation
    megnet/data/graph.py

    Expand distance with Gaussian basis sit at centers and with width 0.5.
    """

    def __init__(self, centers = torch.linspace(0, 5, 100), width=0.5):
        """

        Args:
            centers: (np.array) centers for the Gaussian basis
            width: (float) width of Gaussian basis
        """
        self.centers = centers
        self.width = width

    def convert(self, d: torch.Tensor) -> torch.Tensor:
        """
        expand distance vector d with given parameters
        Args:
            d: (1d array) distance array
        Returns
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        return torch.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width**2)
