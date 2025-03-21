import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path

def create_mask_poly(foil, x_grid, y_grid):
    # Flatten grid coordinates into (N, 2) array of points
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    
    # Create Path object and check containment
    path = matplotlib.path.Path(foil)
    mask = path.contains_points(points)
    
    # Reshape mask back to original grid shape
    return mask.reshape(x_grid.shape)


def plot_heightmap(result, x_grid, y_grid, foil=None, levels=None):
    if foil is None:
        masked_result = result
    else:
        mask = create_mask_poly(foil, x_grid, y_grid)
        masked_result = np.ma.masked_where(mask, result)

    if levels is None:
        levels = np.linspace(np.min(masked_result), np.max(masked_result), 13)

    contour = plt.contour(x_grid, y_grid, result, levels=levels, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    if foil is not None:
        plt.plot(foil[:, 0], foil[:, 1], 'k-', linewidth=2)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.colorbar(contour)
    # plt.gca().set_aspect('equal')
    plt.show()

def plot_heatmap(result, x_grid, y_grid, foil=None, levels=None):
    fig, ax = plt.subplots()

    # Plot the heatmap with correct axis scaling
    heatmap = ax.imshow(result, extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), 
                        origin='upper', cmap='hot', aspect='auto', interpolation='nearest')
    
    # Add colorbar for reference
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Intensity')

    # Axis labels and aspect ratio
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_aspect('equal')

    plt.show()

def plot_velmap(u, v, x_grid, y_grid, foil=None, levels=None):
    fig, ax = plt.subplots()
    result = np.sqrt(u**2 + v**2)

    # Plot the heatmap with correct axis scaling
    heatmap = ax.imshow(result, extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), 
                        origin='lower', cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add colorbar for reference
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Intensity')

    # Axis labels and aspect ratio
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    # ax.set_aspect('equal')

    plt.show()

def plot_foil(data):
    x, y = data[:, 0], data[:, 1]
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(x, y, c=range(len(data)), cmap=cm)
    plt.colorbar(sc)

    # plt.xlim(-.5, 1.5)
    # plt.ylim(-.3, .3)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
