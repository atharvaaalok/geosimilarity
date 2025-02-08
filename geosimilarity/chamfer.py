import torch
import torch.nn as nn
from torch import Tensor


class ChamferLoss(nn.Module):
    r"""Measures the Chamfer Distance between two point sets (curves), which quantifies the average
    closest point distance.

    Given matrices for an input curve :math:`Xc`, and a target curve :math:`Xt` of sizes
    :math:`N \times 2` and :math:`M \times 2`, calculates the Chamfer Distance:
    .. math::
        d_{CD}(Xc, Xt) = \frac{1}{N} \sum_{x \in Xc} \min_{y \in Xt} \|x - y\|^2 +
                          \frac{1}{M} \sum_{y \in Xt} \min_{x \in Xc} \|x - y\|^2
    
    This loss function is useful for shape matching tasks, as it measures how well two
    sets of points align by considering nearest-neighbor distances.
    
    Args:
    
    Shape:
        - Xc: :math:`(N, 2)`
        - Xt: :math:`(M, 2)` (potentially different number of points)
        - Output: scalar.
    
    Examples::
        >>> import geosimilarity as gs
        >>> loss = gs.ChamferLoss()
        >>> Xc = torch.randn(10, 2, requires_grad = True)
        >>> Xt = torch.randn(15, 2)
        >>> output = loss(Xc, Xt)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(ChamferLoss, self).__init__()
    
    
    def forward(self, Xc: Tensor, Xt: Tensor) -> Tensor:
        # Compute pairwise squared Euclidean distances
        dists = torch.cdist(Xc, Xt, p = 2) ** 2  # (N, M)
        
        # Compute minimum distance for each point in Xc to Xt, and vice versa
        min_dist_Xc = torch.min(dists, dim = 1)[0]  # (N,)
        min_dist_Xt = torch.min(dists, dim = 0)[0]  # (M,)
        
        # Compute Chamfer distance
        loss = torch.mean(min_dist_Xc) + torch.mean(min_dist_Xt)
        return loss