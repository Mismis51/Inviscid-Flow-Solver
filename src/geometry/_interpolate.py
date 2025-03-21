import numpy as np
from scipy.interpolate import make_interp_spline

def _interpolate(points, n):
    points = np.array(points)
    if len(points) < 3:
        raise ValueError("At least 3 points required to make a spline")
    
    # Compute cumulative arc length
    diff = np.diff(points, axis=0)
    dist = np.hypot(diff[:,0], diff[:,1])
    cum_dist = np.insert(np.cumsum(dist), 0, 0)
    total_length = cum_dist[-1]
    
    # Create splines parameterized by arc length
    spline_x = make_interp_spline(cum_dist, points[:,0], k=3)
    spline_y = make_interp_spline(cum_dist, points[:,1], k=3)

    # Eliminate any sharp trailing edge
    if np.allclose(points[0], points[-1]):
        # Higher density of points near the edges
        # t = np.linspace(0, 1, n+2)
        # s_new = (1 - np.cos(np.pi * t)) / 2 * total_length

        s_new = np.linspace(0, total_length, n+2)

        x_new = spline_x(s_new)[1:-1]
        y_new = spline_y(s_new)[1:-1]
    else:
        # Higher density of points near the edges
        # t = np.linspace(0, 1, n)
        # s_new = (1 - np.cos(np.pi * t)) / 2 * total_length
        
        s_new = np.linspace(0, total_length, n)

        x_new = spline_x(s_new)
        y_new = spline_y(s_new)
    
    return np.column_stack((x_new, y_new))
