import torch
from shape_svg.svg_extract_xy import svg_extract_xy


def circle(num_pts: int) -> torch.Tensor:
    theta = torch.linspace(0, 2 * torch.pi, num_pts)
    X = torch.stack([torch.cos(theta), torch.sin(theta)], dim = 1)
    return X


def square(num_pts: int) -> torch.Tensor:
    # Generate points on the unit circle first and then map them to a square
    theta = torch.linspace(0, 2 * torch.pi, num_pts)

    x, y = torch.cos(theta), torch.sin(theta)
    s = torch.maximum(torch.abs(x), torch.abs(y))
    x_sq, y_sq = x/s, y/s

    X = torch.stack([x_sq, y_sq], dim = 1)
    return X


def stanford_bunny(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('stanford_bunny.svg', num_pts = num_pts)
    return X

def heart(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('heart.svg', num_pts = num_pts)
    return X


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = square(num_pts = 100)
    plt.plot(X[:, 0], X[:, 1])
    plt.show()