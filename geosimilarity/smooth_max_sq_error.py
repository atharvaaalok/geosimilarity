import torch
import torch.nn as nn
from torch import Tensor


class SmoothMaxSquaredErrorLoss(nn.Module):
    r"""Measures a Smooth Maximum of Squared Errors, using the LogSumExp function to approximate the
    maximum while maintaining differentiability.

    Given matrices for an input curve :math:`Xc`, and a target curve :math:`Xt` of same size
    :math:`N \times 2`, calculates:
    .. math:
        L(Xc, Xt) = \tau \log \sum_{i=0}^{N-1}
            \exp \left( \frac{(Xc[i, 0] - Xt[i, 0])^2 + (Xc[i, 1] - Xt[i, 1])^2}{\tau} \right)
    
    where :math:`\tau` is a temperature parameter that controls the sharpness of the smooth maximum.
    Lower values of :math:`\tau` make the approximation closer to the hard maximum.
    
    Args:
        tau (float, optional): Temperature parameter controlling the smoothness of the maximum.
    
    Shape:
        - Xc: :math:`(N, 2)`
        - Xt: :math:`(N, 2)`, same size as Xc.
        - Output: scalar.
    
    Examples::
        >>> import geosimilarity as gs
        >>> loss = gs.SmoothMaxSquaredErrorLoss(tau = 0.1)
        >>> Xc = torch.randn(10, 2, requires_grad = True)
        >>> Xt = torch.randn(10, 2)
        >>> output = loss(Xc, Xt)
        >>> output.backward()
    """

    def __init__(self, tau: float = 1.0) -> None:
        super(SmoothMaxSquaredErrorLoss, self).__init__()
        self.tau = tau
    
    def forward(self, Xc: Tensor, Xt: Tensor) -> Tensor:
        corresponding_sq_error = torch.sum((Xc - Xt) ** 2, dim = 1)
        loss = self.tau * torch.logsumexp(corresponding_sq_error / self.tau, dim = 0)
        return loss