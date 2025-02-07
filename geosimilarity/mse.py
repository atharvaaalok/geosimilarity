import torch
import torch.nn as nn
from torch import Tensor


class MSELoss(nn.Module):
    r"""Measures the Mean Squared Error (MSE), i.e. squared L2 norm between each element in the
    input and the target curves.

    Given matrices for an input curve :math:`Xc`, and a target curve :math:`Xt` of same size
    :math:`N \times 2`, calculates:
    .. math:
        L(Xc, Xt) = \frac{1}{N} \sum_{i = 0}^{N - 1}
            \left( (Xc[i, 0] - Xt[i, 0])^2 + (Xc[i, 1] - Xt[i, 1])^2 \right)
    
    Args:
    
    Shape:
        - Xc: :math:`(N, 2)`
        - Xt: :math:`(N, 2)`, same size as Xc.
        - Output: scalar.
    
    Examples::
        >>> import geosimilarity as gs
        >>> loss = gs.MSELoss()
        >>> Xc = torch.randn(10, 2, requires_grad = True)
        >>> Xt = torch.randn(10, 2)
        >>> output = loss(Xc, Xt)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(MSELoss, self).__init__()
    
    
    def forward(self, Xc: Tensor, Xt: Tensor) -> Tensor:
        loss = torch.mean((Xc - Xt) ** 2)
        return loss