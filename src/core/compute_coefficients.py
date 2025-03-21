import numpy as np
import copy

def compute_lift(geometry, gammas):
    integrand = (gammas + np.roll(gammas, -1)) * geometry.ds / 2
    return 2 * np.sum(integrand)

def compute_coefficients(geometry, gammas):
        # Pressure coefficient distribution.
    cp = 1 - gammas**2

    # Initialize force and moment coefficients.
    cd = 0  # drag (x-direction)
    cl = 0  # lift (y-direction)
    cm = 0  # moment

    ds = geometry.ds

    # Get the rotated vertices (points in physical space).
    vertex = geometry.get_rotated_vertex()

    # Create a copy to compute normals based on rotated vertices.
    temp_geom = copy.deepcopy(geometry)
    temp_geom.vertex = vertex
    temp_geom._compute_parameters()
    normals = temp_geom.normal

    nx, ny = normals[:, 0], -normals[:, 1]

    n_panels = len(cp)
    for i, cp_val in enumerate(cp):
        cp_next = cp[(i + 1) % n_panels]
        cp_avg = (cp_val + cp_next) / 2

        # Accumulate drag and lift from each panel.
        cd += cp_avg * ds[i] * nx[i]
        cl += cp_avg * ds[i] * ny[i]

        # Compute the midpoint of the panel (assuming a closed geometry).
        mid = (vertex[i] + vertex[(i + 1) % n_panels]) / 2

        # Moment: r x F, where r is the midpoint.
        # F = cp_avg * ds[i] * (nx[i], ny[i])
        # So, moment contribution = cp_avg * ds[i] * (mid_x * ny[i] - mid_y * nx[i])
        # Calculate the moment contribution about the quarter chord.
        # Shift the x-coordinate by subtracting the quarter chord location.
        quarter_chord_x = 1 / 4.0
        cm += cp_avg * ds[i] * ((mid[0] - quarter_chord_x) * ny[i] - mid[1] * nx[i])

    return cl, cd, cm
