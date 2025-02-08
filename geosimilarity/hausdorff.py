import torch
import torch.nn as nn
from torch import Tensor


class HausdorffLoss(nn.Module):
    r"""Measures the discrete Hausdorff distance between two curves.

    Given matrices for an input curve :math:`Xc`, and a target curve :math:`Xt` of sizes
    :math:`N \times 2` and :math:`M \times 2`, calculates:
    .. math::
        d_H(Xc, Xt) = \max \left( \max_{x \in Xc} \min_{y \in Xt} \|x - y\|, 
                                   \max_{y \in Xt} \min_{x \in Xc} \|x - y\| \right)
    
    Args:
    
    Shape:
        - Xc: :math:`(N, 2)`
        - Xt: :math:`(M, 2)` (potentially different number of points)
        - Output: scalar.
    
    Examples::
        >>> import geosimilarity as gs
        >>> loss = gs.HausdorffLoss()
        >>> Xc = torch.randn(10, 2, requires_grad = True)
        >>> Xt = torch.randn(15, 2)
        >>> output = loss(Xc, Xt)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(HausdorffLoss, self).__init__()
    
    
    def forward(self, Xc: Tensor, Xt: Tensor) -> Tensor:
        # Compute pairwise Euclidean distances
        dists = torch.cdist(Xc, Xt, p = 2)  # (N, M)
        
        # Compute minimum distance for each point in Xc to Xt, and vice versa
        # torch.min() returns tuple of values, indices. Use [0] indexing to extract only values
        min_dist_Xc = torch.min(dists, dim = 1)[0]  # (N,)
        min_dist_Xt = torch.min(dists, dim = 0)[0]  # (M,)
        
        # Compute the Hausdorff distance
        loss = torch.max(torch.max(min_dist_Xc), torch.max(min_dist_Xt))
        return loss