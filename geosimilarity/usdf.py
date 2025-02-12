import torch
import torch.nn as nn
from torch import Tensor


class USDFLoss(nn.Module):
    r"""Computes the difference between the UnSigned Distance Fields (USDF) of two curves.
    
    Given matrices for an input curve :math:`Xc`, and a target curve :math:`Xt` of sizes
    :math:`N \times 2` and :math:`M \times 2`, generates a grid over the bounding region of the
    curves and calculates the unsigned distance field for both. The difference between these fields,
    measured by Mean Squared Error is used as the loss.
    
    Args:
        grid_res (int, optional): Grid resolution per dimension (default: 32).
    
    Shape:
        - Xc: :math:`(N, 2)`
        - Xt: :math:`(M, 2)` (potentially different number of points)
        - Output: scalar.
    
    Examples::
        >>> import geosimilarity as gs
        >>> loss = gs.USDFLoss(grid_res = 100)
        >>> Xc = torch.randn(10, 2, requires_grad = True)
        >>> Xt = torch.randn(15, 2)
        >>> output = loss(Xc, Xt)
        >>> output.backward()
    """

    def __init__(self, grid_res: int = 32) -> None:
        super(USDFLoss, self).__init__()
        self.grid_res = grid_res
    
    
    def forward(self, Xc: Tensor, Xt: Tensor) -> Tensor:
        # Generate a regular grid with given resolution
        x_min, y_min = torch.min(torch.cat([Xc, Xt], dim = 0), dim = 0)[0]
        x_max, y_max = torch.max(torch.cat([Xc, Xt], dim = 0), dim = 0)[0]
        grid = generate_grid(x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max,
                             grid_res = self.grid_res)
        
        # Compute USDF for Xc
        usdf_Xc = compute_unsigned_distance_field(Xc, grid)
        # Compute USDF for Xt
        usdf_Xt = compute_unsigned_distance_field(Xt, grid)

        # Compute difference between the USDFs
        loss = torch.mean((usdf_Xc - usdf_Xt) ** 2)
        return loss


def generate_grid(x_min: float, x_max: float, y_min: float, y_max: float, grid_res: int) -> Tensor:
    r"""Generates a regular Cartesian grid of specified resolution given the coordinates of bounding
    box.

    Args:
        x_min (float): minimum x for the grid bounding box.
        x_max (float): maximum x for the grid bounding box.
        y_min (float): minimum y for the grid bounding box.
        y_max (float): maximum y for the grid bounding box.
        grid_res (int): Grid resolution per dimension.
    
    Returns:
        grid (Tensor): A matrix of dimensions (grid_res ** 2, 2) with coordinates of grid points in
            each row (x, y).
    """
    
    x_vals = torch.linspace(x_min, x_max, grid_res)
    y_vals = torch.linspace(y_min, y_max, grid_res)

    grid_x, grid_y = torch.meshgrid(x_vals, y_vals, indexing = "xy")
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim = 1)  # (grid_res**2, 2)
    return grid


def compute_unsigned_distance_field(X: Tensor, grid: Tensor) -> Tensor:
    """Computes the UnSigned Distance Field (USDF) for a given point set on a grid.

    Args:
        X (Tensor): A matrix of dimension (N, 2) with each row corresponding to the (x, y)
            coordinates of specified point set.
        grid (Tensor): A grid matrix of dimension (G, 2) with each row corresponding to the (x, y)
            coordinates of grid points in space.
    
    Returns:
        usdf (Tensor): An unsigned distance field matrix (G, 2) calculated at the grid points
            specified.
    """
    
    # Compute pairwise distances (G, N)
    dists = torch.cdist(grid, X, p = 2)
    # Take the minimum distance to define the USDF
    usdf = torch.min(dists, dim = 1)[0]
    return usdf