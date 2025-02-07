import torch

from .utils import LinearSpline, plot_curves
from .shapes import square


# Get a target curve
Xt = square(num_pts = 100)

# Create a linear spline object
spline = LinearSpline(num_control_pts = 10)

# Generate points on the spline
Xc = spline(num_pts = Xt.shape[0])


# Plot and compare the candidate and the target curves
plot_curves(Xc, Xt)