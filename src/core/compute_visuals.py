import numpy as np

def compute_streamlines(flows, x_grid, y_grid):
    # Iterate over the grid and compute the influence of each source
    result_grid = np.array(sum([flow.streamline(x_grid, y_grid) for flow in flows]))
            
    return result_grid

def compute_velocities(flows, x_grid, y_grid):
    u_grid, v_grid = flows[0].velocity(x_grid, y_grid)

    # Iterate over the grid and compute the influence of each source
    for flow in flows[1:]:
        u, v = np.sum(flow.velocity(x_grid, y_grid), axis=-1)
        u_grid += u
        v_grid += v

    return u_grid, v_grid