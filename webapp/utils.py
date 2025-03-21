import numpy as np
from src.flows import freestream, linear_vortex
from src import compute_streamlines, compute_velocities

def compute_flow(visualization_mode, vortex_strengths, airfoil_points, resolution=200):
    """
    Compute flow field data for visualization.
    
    Args:
        visualization_mode: 'streamlines' or 'velocities' 
        vortex_strengths: Array of vortex strengths along airfoil
        airfoil_points: Coordinates defining the airfoil geometry
        resolution: Grid resolution for flow computation (default: 200)
    
    Returns:
        Tuple of (x_grid, y_grid, flow_data) for plotting
    """
    # Initialize flow components with freestream
    flow_components = [freestream(u_inf=1)]
    
    # Add linear vortex components for each airfoil segment
    num_segments = len(vortex_strengths)
    for i in range(num_segments):
        start_point = airfoil_points[i]
        end_point = airfoil_points[(i + 1) % num_segments]
        current_strength = vortex_strengths[i]
        next_strength = vortex_strengths[(i + 1) % num_segments]
        
        flow_components.append(
            linear_vortex(current_strength, next_strength, start_point, end_point)
        )

    # Create computation grid
    x_points = np.linspace(-0.5, 1.5, resolution)  # Normalized airfoil coordinates
    y_points = np.linspace(-1.0, 1.0, resolution)
    x_grid, y_grid = np.meshgrid(x_points, y_points)

    # Compute requested flow visualization
    if visualization_mode == 'streamlines':
        flow_data = compute_streamlines(flow_components, x_grid, y_grid)
    else:
        velocity_x, velocity_y = compute_velocities(flow_components, x_grid, y_grid)
        flow_data = np.sqrt(velocity_x**2 + velocity_y**2)  # Velocity magnitude

    return x_grid, y_grid, flow_data
