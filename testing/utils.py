import torch
from typing_extensions import Self


class LinearSpline():
    def __init__(self, num_control_pts: int) -> None:
        self.num_control_pts = num_control_pts

        # Generate initial set of control points on unit circle
        # Do not make the first and last point overlap, appropriately choose final angle to have
        # equal spacing between the generated points
        last_angle = 2 * torch.pi * (num_control_pts - 1)/(num_control_pts)
        theta = torch.linspace(0, last_angle, num_control_pts)
        self.CP = torch.stack((torch.cos(theta), torch.sin(theta)), dim = 1)
        self.CP.requires_grad_()
    

    def to(self, device: str) -> Self:
        self.CP = self.CP.to(device)
        return self
    

    def generate(self, num_pts: int) -> torch.Tensor:
        # Extend the control points by appending the first control point
        # This ensures that the segment from the last to the first point is included
        CP_ext = torch.cat([self.CP, self.CP[:1]], dim = 0)

        # There are num_control_pts segments now (one per edge of the closed shape)
        # Create a parameter t uniformly spaced in [0, num_control_pts)
        t = torch.linspace(0, self.num_control_pts, steps = num_pts + 1,
                           device = self.CP.device, dtype = self.CP.dtype)[:-1]
        
        # For each t, determine indices of the starting control point for each segment
        idx = torch.floor(t).long()
        # Compute the fractional part for interpolation in each segment
        alpha = (t - idx.to(self.CP.dtype)).unsqueeze(1)
        
        # Perform linear interpolation between CP_ext[idx] and CP_ext[idx+1]
        points = (1 - alpha) * CP_ext[idx] + alpha * CP_ext[idx + 1]
        return points
    

    def __call__(self, num_pts: int) -> torch.Tensor:
        return self.generate(num_pts)