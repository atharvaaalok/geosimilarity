import torch
import torch.nn as nn
from torch import Tensor


class MaxSquaredErrorLoss(nn.Module):
    r"""Measures the Maximum Squared Error, i.e. the maximum squared L2 norm between corresponding
    elements in the input and the target curves.

    Given matrices for an input curve :math:`Xc`, and a target curve :math:`Xt` of same size
    :math:`N \times 2`, calculates:
    .. math:
        L(Xc, Xt) = \max_{i \in \{0, \ldots, N-1\}}
            \left( (Xc[i, 0] - Xt[i, 0])^2 + (Xc[i, 1] - Xt[i, 1])^2 \right)
    
    Args:
    
    Shape:
        - Xc: :math:`(N, 2)`
        - Xt: :math:`(N, 2)`, same size as Xc.
        - Output: scalar.
    
    Examples::
        >>> import geosimilarity as gs
        >>> loss = gs.MaxSquaredErrorLoss()
        >>> Xc = torch.randn(10, 2, requires_grad = True)
        >>> Xt = torch.randn(10, 2)
        >>> output = loss(Xc, Xt)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(MaxSquaredErrorLoss, self).__init__()
    
    
    def forward(self, Xc: Tensor, Xt: Tensor) -> Tensor:
        corresponding_sq_error = torch.sum((Xc - Xt) ** 2, dim = 1)
        loss = torch.max(corresponding_sq_error)
        return loss