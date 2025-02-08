import torch
import torch.nn as nn
from torch import Tensor


class SmoothHausdorffLoss(nn.Module):
    r"""Measures a differentiable approximation of the Hausdorff distance between two curves, using
    Smooth Maximum (LogSumExp) and Smooth Minimum (SoftMin) to maintain differentiability.

    Given matrices for an input curve :math:`Xc`, and a target curve :math:`Xt` of sizes
    :math:`N \times 2` and :math:`M \times 2`, calculates an approximation of the discrete Hausdorff
    distance:
    .. math::
        d_H(Xc, Xt) \approx \tilde{\max} \left(
            \tilde{\max}_{x \in Xc} \tilde{\min}_{y \in Xt} \|x - y\|,
            \tilde{\max}_{y \in Xt} \tilde{\min}_{x \in Xc} \|x - y\|
            \right)
    
    Using smooth approximations:
    .. math::
        \tilde{\min}(x) = - \tau \log \sum \exp(-x / \tau)
    
        \tilde{\max}(x) = \tau \log \sum \exp(x / \tau)
    
    where :math:`\tau` is a temperature parameter that controls the smoothness of the min/max
    approximations. Lower values of :math:`\tau` make the approximation closer to the true Hausdorff
    distance.
    
    Args:
        tau (float, optional): Temperature parameter controlling the smoothness of the
        approximations.
    
    Shape:
        - Xc: :math:`(N, 2)`
        - Xt: :math:`(M, 2)` (potentially different number of points)
        - Output: scalar.
    
    Examples::
        >>> import geosimilarity as gs
        >>> loss = gs.SmoothHausdorffLoss(tau = 0.1)
        >>> Xc = torch.randn(10, 2, requires_grad = True)
        >>> Xt = torch.randn(15, 2)
        >>> output = loss(Xc, Xt)
        >>> output.backward()
    """

    def __init__(self, tau: float = 1.0) -> None:
        super(SmoothHausdorffLoss, self).__init__()
        self.tau = tau
    

    def forward(self, Xc: Tensor, Xt: Tensor) -> Tensor:
        # Compute pairwise Euclidean distances
        dists = torch.cdist(Xc, Xt, p = 2)  # (N, M)
        
        # Smooth minimum (SoftMin) along each row and column
        softmin_dist_Xc = -self.tau * torch.logsumexp(-dists / self.tau, dim = 1)  # (N,)
        softmin__dist_Xt = -self.tau * torch.logsumexp(-dists / self.tau, dim = 0)  # (M,)
        
        # Smooth maximum (LogSumExp) over each set
        softmax_Xc = self.tau * torch.logsumexp(softmin_dist_Xc / self.tau, dim = 0)
        softmax_Xt = self.tau * torch.logsumexp(softmin__dist_Xt / self.tau, dim = 0)

        loss = self.tau * torch.logsumexp(torch.stack([softmax_Xc, softmax_Xt]) / self.tau, dim = 0)
        return loss