import numpy as np

def _normalize_and_center(vertex):
    # Compute x and y ranges
    x_min, x_max = np.min(vertex[:, 0]), np.max(vertex[:, 0])
    y_min, y_max = np.min(vertex[:, 1]), np.max(vertex[:, 1])

    # Scaling factor based on x-range
    scale_factor = 1 / (x_max - x_min)

    # Scale x to [0, 1]
    x_scaled = (vertex[:, 0] - x_min) * scale_factor

    # Center y and scale proportionally
    y_center = (y_min + y_max) / 2
    y_scaled = (vertex[:, 1] - y_center) * scale_factor

    # Combine scaled points
    return np.column_stack((x_scaled, y_scaled))

def _rotate_around_ahalf(matrix, angle):
    rotation_matrix = np.array((
        (np.cos(angle), -np.sin(angle), .5 * (1 - np.cos(angle))),
        (np.sin(angle), np.cos(angle), -.5 * np.sin(angle)),
        (0, 0, 1)
    ))

    rotated_array = rotation_matrix @ np.vstack((matrix.T, np.ones((len(matrix)))))
    return rotated_array.T[:, :-1]