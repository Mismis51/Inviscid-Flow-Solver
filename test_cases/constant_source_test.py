import numpy as np
import src
import matplotlib.pyplot as plt


p1 = np.vstack(((0, 0), (1, 0)))
p2 = np.array(((1, 0), (2, 1)))

a_s = np.array((2, 2))
b_s = np.array((2, 0))


flows = [
    # src.flows.linear_vortex(a_s, b_s, p1, p2)
    src.flows.constant_source(1, (-.5, -.5), (.5, .5))
    ]

nx, ny = (100, 100)
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
x_grid, y_grid = np.meshgrid(x, y)

result = src.compute_streamlines(flows, x_grid, y_grid)
src.plot_heightmap(result, x_grid, y_grid, levels=None)

u, v = flows[0].velocity(x_grid, y_grid)
im = plt.streamplot(x_grid, y_grid, np.sum(u, axis=-1), np.sum(v, axis=-1), color='k', cmap='viridis')
plt.show()