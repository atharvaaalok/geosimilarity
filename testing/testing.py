import torch

from .utils import LinearSpline, plot_curves, automate_training
from .shapes import square
from ..geosimilarity import MSELoss


# Get a target curve
Xt = square(num_pts = 100)

# Create a linear spline object
spline = LinearSpline(num_control_pts = 50)

# Generate points on the spline
Xc = spline(num_pts = Xt.shape[0])

# Plot and compare the candidate and the target curves
plot_curves(Xc, Xt)


# Choose a particular loss function
loss_fn = MSELoss()

# Train the spline control points to fit the target curve
automate_training(spline, num_candidate_pts = Xt.shape[0], Xt = Xt, loss_fn = loss_fn,
                  epochs = 100, print_cost_every = 20, learning_rate = 0.1)

# Plot and compare the fitted candidate and the target curves
Xc = spline(num_pts = Xt.shape[0])
plot_curves(Xc, Xt)