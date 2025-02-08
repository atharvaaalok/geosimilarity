# Code inspired from: https://github.com/spotify/coordinator

import numpy as np
import torch
from typing import List, Tuple
from svg.path import parse_path
from xml.dom import minidom
import os


def get_total_length_all_paths(paths: List[str]) -> float:
    """ Calculate the total length of all paths."""
    total_length = 0
    for path in paths:
        parsed_path = parse_path(path)
        total_length += parsed_path.length()
    return total_length


def polygonize(
    path: str,
    num_points: int,
    scale: float,
    translate_x: float,
    translate_y: float,
) -> List[Tuple[float, float]]:
    """Convert an SVG path to a set of evenly spaced coordinates."""
    parsed_path = parse_path(path)
    points = []
    for i in range(num_points):
        t = i / max(num_points - 1, 1)  # Normalize to [0, 1]
        point = parsed_path.point(t)
        x = point.real * scale + translate_x
        y = point.imag * scale + translate_y
        points.append((x, y))
    return points


def paths_to_coords(
    paths: List[str],
    scale: float,
    num_points: int,
    translate_x: float,
    translate_y: float,
) -> List[Tuple[float, float]]:
    """Distribute a specified number of points across multiple paths and return their
    coordinates."""
    total_length_all_paths = get_total_length_all_paths(paths)
    running_points_total = 0
    separate_paths_coords_collection = []

    for index, path in enumerate(paths):
        if index + 1 == len(paths):
            # Assign remaining points to the last path
            points_for_path = num_points - running_points_total
        else:
            parsed_path = parse_path(path)
            path_length = parsed_path.length()
            points_for_path = round(num_points * path_length / total_length_all_paths)
            running_points_total += points_for_path

        # Generate points for this path
        separate_paths_coords_collection.extend(
            polygonize(path, points_for_path, scale, translate_x, translate_y)
        )

    return separate_paths_coords_collection


def svg_extract_xy(filename: str, num_pts: int) -> torch.Tensor:
    # Load SVG paths from an SVG file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    svg_file_path = os.path.join(current_dir, filename)
    doc = minidom.parse(svg_file_path)
    path_strings = [path.getAttribute("d") for path in doc.getElementsByTagName("path")]
    doc.unlink()

    # Parameters
    scale = 1.0
    num_points = num_pts
    translate_x = 0.0
    translate_y = 0.0

    # Generate coordinates
    coords = paths_to_coords(path_strings, scale, num_points, translate_x, translate_y)

    # Generated points are mirrored about the x-axis, flip to get the right figure
    X = np.array(coords)
    X[:, 1] = -X[:, 1]

    # Center at origin
    centroid = np.mean(X, axis=0)
    X = X - centroid

    # Scale the points to fit within a unit box [-1, 1]
    max_abs = np.max(np.abs(X))
    X = X / max_abs

    # Convert to torch.float32 to be compliant with deep learning models
    X = torch.from_numpy(X).to(torch.float32)

    return X


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X = svg_extract_xy(filename =  'stanford_bunny.svg', num_pts = 100)

    plt.plot(X[:, 0], X[:, 1], marker = 'o')
    plt.axis('equal')
    plt.show()